# Copyright 2019-2025 AstroLab Software
# Author: Sergey Karpov
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from fink_utils.tg_bot.utils import msg_handler_tg, escape

from fink_filters.tester import spark_unit_tests

import numpy as np
import pandas as pd
import os
import io
import time

from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy import units as u

import xgboost as xgb

import matplotlib.pyplot as plt

# Slack
from slack_sdk import WebClient

# Telegram

from fink_filters.ztf.filter_early_tde_candidates import prefilter, lcs


def find_candidates(
    data,
    window=100,
    hist_window=100,
    nsamples=1000,
    classifiers=None,
    plot_lc=False,
    skip_classified=True,
):
    """Find early TDE candidates in pre-filtered dataframe.

    Parameters
    ----------
    data: DataFrame
        pre-filtered data produced by prefilter.prefilter_alerts()
    window: int
        Fitting window size, days
    hist_window: int
        Historical window size, days
    nsamples: int
        Number of feature samples to generate according to fitted parameter covariances
    classifiers: list
        List containing the classifiers to apply to the features
    plot_lc: bool
        Whether to include the light curve plots in the output
    skip_classified: bool
        Whether to skip the candidates with already available TNS classification

    Returns
    -------
    out: Pandas DataFrame
         Pandas DataFrame with scores and parameters for the selected candidates
    """
    candidates = []

    if classifiers is None:
        # Load default classifiers
        classifiers = []

        curdir = os.path.dirname(os.path.abspath(__file__))
        for _ in ["model_nuclear.ubj", "model_broad.ubj"]:
            clf = xgb.XGBClassifier()
            clf.load_model(os.path.join(curdir, "data", _))
            classifiers.append(clf)

    for _, cand in data.iterrows():
        if skip_classified:
            # Skip the object if it has TNS classification, and it is not TDE
            types = lcs.get_tns_info(ra=cand["ra"], dec=cand["dec"], return_types=True)
            if len(types) and "TDE" not in types and "TDE-He" not in types:
                continue

        # Use Fink API, with some points potentially missing
        pdf = lcs.request_lc(cand["objectId"])
        if pdf is None:
            continue
        # Merge with detections from alert, if they are missing
        sub = lcs.get_lc(cand, prefix="c")
        pdf = (
            pd.concat([pdf, sub[~np.isin(sub["i:jd"], pdf["i:jd"])]])
            .sort_values("i:jd")
            .reset_index(drop=True)
        )

        # Fitting window
        jd_max = cand["jd"]
        jd_min = jd_max - window
        idx0 = (pdf["i:jd"] <= jd_max) & (pdf["i:jd"] >= jd_min)  # window

        sub = pdf[idx0]

        # Pre-window history
        hist_idx = (pdf["i:jd"] <= jd_min) & (
            pdf["i:jd"] > jd_min - hist_window
        )  # History window prior to the window above
        num_hist_detections = np.sum(
            hist_idx & np.isfinite(pdf["FLUXCAL"])
        )  # Detections in historical window
        # num_hist_limits = np.sum(
        #     hist_idx & ~np.isfinite(pdf["FLUXCAL"])
        # )  # Upper limits

        # All prior detections
        prehist_idx = pdf["i:jd"] <= jd_min  # Everything prior to the window above
        # num_prehist_detections = np.sum(
        #     prehist_idx & np.isfinite(pdf["FLUXCAL"])
        # )  # Detections prior to window
        num_prehist_negatives = np.sum(
            prehist_idx & np.isfinite(pdf["FLUXCAL"]) & (pdf["FLUXCAL"] < 0)
        )  # Negative detections prior to window
        # num_prehist_limits = np.sum(
        #     prehist_idx & ~np.isfinite(pdf["FLUXCAL"])
        # )  # Upper limits

        # Remove redundant upper limits within window
        sub = lcs.cleanup_limits(sub)

        # WARNING: Discard upper limits, for now!!!
        # sub = sub[np.isfinite(sub['FLUXCAL'])]

        # Number of detections and limits
        idx = np.isfinite(sub["FLUXCAL"])

        # Unused variables
        # num_detections = np.sum(idx)
        # num_negatives = np.sum(idx & (sub["FLUXCAL"] < 0))
        # num_limits = np.sum(~idx)

        # Initial cuts
        if num_hist_detections > 0 or num_prehist_negatives > 1:
            # Either historical detections, or more than one pre-historical negative
            continue

        # TODO: re-check rising / not fading criteria?..

        # Extract Rainbow features - best fit and sampled according to covariances
        res1 = lcs.extract_features(sub, nsamples=nsamples)
        # Initial quality cuts for the fit
        if res1 is None:
            continue

        # Actual features
        res = {}
        for i, name in enumerate(lcs.feature.names):
            if name == "T":
                # Normalize temperature name
                name = "temperature"
            res[name] = np.array([res1["params"][i]] + [_[i] for _ in res1["samples"]])
            res["e_" + name] = res1["errors"][i]

            if name not in ["reference_time"]:
                res["snr_" + name] = np.abs(res1["params"][i] / res1["errors"][i])

        res["r_chisq"] = res1["params"][-1]
        res["rel_reference_time"] = res["reference_time"] - jd_max
        res["norm_rel_reference_time"] = res["rel_reference_time"] / res["rise_time"]
        res["distnr"] = np.nanmin(pdf["i:distnr"])

        # Features quality cuts
        if (
            res["r_chisq"] > 10
            or res["e_reference_time"] > 100
            or res["norm_rel_reference_time"][0] > 1
            or res["norm_rel_reference_time"][0] < -10
            or
            # res['snr_amplitude'] < 1.5 or
            res["snr_rise_time"] < 1.5
            or res["snr_temperature"] < 1.5
        ):
            continue

        candidate = {
            "objectId": cand["objectId"],
            "cand": cand,
            "res": res,
            "best_score": [],
            "frac_scores": [],
            "valid": True,
        }

        for clf in classifiers:
            # Select features that are used in classifier
            features = pd.DataFrame(res)[clf.feature_names_in_]

            # Run the classifier
            scores = clf.predict(features)  # Best fit + sampled
            best_score = scores[0]  # Best fit score
            frac_scores = np.sum(scores) / len(
                scores
            )  # Fraction of samples that are scored positive

            candidate["best_score"].append(best_score)
            candidate["frac_scores"].append(frac_scores)

            if not best_score and frac_scores < 0.1:
                candidate["valid"] = False

        if candidate["valid"] and plot_lc:
            # Lightcurve plotting
            plt.figure(figsize=(14, 10))

            # Upper panel, lightcurves
            plt.subplot(2, 1, 1)
            lcs.plot_lc(pdf, show_zero=True, show_limits=False, ms=10)

            # Request DR photometry from SNAD
            snad = lcs.request_lc_snad(cand["ra"], cand["dec"], sr_arcsec=1.5)
            if snad is not None:
                for c, fid in ((lcs.COLORS_ZTF[1], 1), (lcs.COLORS_ZTF[2], 2)):
                    idx = (
                        (pdf["i:fid"] == fid)
                        & (np.isfinite(pdf["i:magnr"]))
                        & (pdf["i:distnr"] < 1.5)
                    )
                    if np.sum(idx):
                        # Baseline template flux
                        flux0 = 10 ** (11 - 0.4 * np.mean(pdf["i:magnr"][idx]))
                    else:
                        flux0 = 0

                    idx = snad["i:fid"] == fid
                    plt.plot(
                        snad["i:jd"][idx] - 2400000.5,
                        snad["FLUXCAL"][idx] - flux0,
                        ".",
                        ms=5,
                        alpha=0.2,
                        color=c,
                    )

            plt.axvline(jd_min - 2400000.5, ls="--", color="black", alpha=0.5)
            plt.axvline(jd_max - 2400000.5, ls="--", color="black", alpha=0.5)

            plt.title(
                cand["objectId"]
                + "  :  "
                + cand["classification"]
                + "  :  "
                + lcs.get_tns_info(ra=cand["ra"], dec=cand["dec"])
            )
            plt.margins(0.02, 0.05)

            plt.legend(loc="upper left")

            # Lower panel, Rainbow fit
            plt.subplot(2, 1, 2)
            lcs.plot_features(
                sub,
                res1,
                show_zero=True,
                show_limits=True,
                prior=window,
                ms=10,
                extra=[
                    "",
                    f"distnr = {res['distnr']:.2f}",
                    f"rel_reference_time = {res['rel_reference_time'][0]:.1f}",
                    f"norm_rel_reference_time = {res['norm_rel_reference_time'][0]:.1f}",
                ],
            )
            title = "best fit score = "
            for _ in candidate["best_score"]:
                title += f"{_:.0f} "
            title += " :  fraction = "
            for _ in candidate["frac_scores"]:
                title += f"{_:.2f} "

            plt.title(title)
            plt.margins(0.02, 0.05)

            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            candidate["lc"] = buf

        if candidate["valid"]:
            # TODO: return all candidates, to report probabilities?..
            candidates.append(candidate)

    return candidates


def early_tde_candidates(
    df,
    prefiltered=None,
    send_to_tg=False,
    tg_channel="@fink_early_tdes",
    send_to_slack=False,
    slack_channel="bot_early_tde",
    sleep=5,
):
    """Get early TDE scores, and send the candidates out.

    Notes
    -----
    Notifications can be sent to a Slack or Telegram channels.

    Parameters
    ----------
    df : Spark DataFrame
    send_to_tg: optional, boolean
        If true, send message to Telegram. Default is False
    tg_channel: str
        If `send_to_tg` is True, `channel_id` is the name of the
        Telegram channel.
    send_to_slack: optional, boolean
        If true, send message to Slack. Default is False
    slack_channel: str
        If `send_to_slack` is True, `channel_name` is the name of the
        Slack channel.
    sleep: int
        Sleep time between sending candidates, in seconds. Default is 5.


    Returns
    -------
    out: Pandas DataFrame
        Pandas DataFrame with scores and parameters for the selected candidates
    """
    if prefiltered is None:
        # Apply preliminary filtering
        prefiltered = prefilter.prefilter_alerts(df)

    data = prefiltered.toPandas()
    data = data.sort_values(
        ["objectId", "jd"], inplace=False
    )  # Report alerts in proper order, grouped by objectId

    candidates = find_candidates(data, plot_lc=True, skip_classified=True)

    if send_to_slack:
        slack_client = WebClient(os.environ["EARLY_TDE_SLACK_TOKEN"])

    slack_msgs = []
    tg_msgs = []

    for candidate in candidates:
        if not candidate["valid"]:
            continue

        cand = candidate["cand"]

        # Useful links
        ps1_link = f"http://alasky.u-strasbg.fr/hips-image-services/hips2fits?hips=CDS%2FP%2FPanSTARRS%2FDR1%2Fcolor-z-zg-g&ra={cand['ra']}&dec={cand['dec']}&width=256&height=256&fov={30 / 3600}&projection=TAN&coordsys=icrs&rotation_angle=0.0&format=jpg"

        tns_link = f"https://www.wis-tns.org/search?ra={cand['ra']}&decl={cand['dec']}&radius=5&coords_unit=arcsec"
        simbad_link = f"http://simbad.u-strasbg.fr/simbad/sim-coo?Coord={cand['ra']}%20{cand['dec']}&Radius=0.08"
        snad_link = f"https://ztf.snad.space/search/{cand['ra']}%20{cand['dec']}/5"
        aavso_link = f"https://www.aavso.org/vsx/index.php?view=results.get&coords={cand['ra']}{'-' if cand['dec'] < 0 else '+'}{cand['dec']}&format=d&size=0.1"

        # Metadata
        sc = SkyCoord(cand["ra"], cand["dec"], unit="deg")
        sc_equ = (
            sc.ra.to_string(unit=u.hourangle, sep=" ", precision=2, pad=True)
            + " "
            + sc.dec.to_string(
                unit=u.deg, sep=" ", precision=1, pad=True, alwayssign=True
            )
        )
        sc_gal = sc.galactic.to_string()

        time_str = Time(cand["jd"], format="jd").strftime("%Y-%m-%d %H:%M:%S")
        ndet = np.sum(~pd.isnull(cand["cmagpsf"]))  # noqa: PD003
        nlim = np.sum(pd.isnull(cand["cmagpsf"]))  # noqa: PD003

        tns_info = lcs.get_tns_info(ra=cand["ra"], dec=cand["dec"])

        if send_to_slack:
            # Upload the LC to Slack
            candidate["lc"].seek(0)
            result_lc = slack_client.files_upload_v2(
                file_uploads=[
                    {"file": candidate["lc"], "title": "lightcurve"},
                ]
            )
            time.sleep(3)

            id_lc = result_lc.data["files"][0]["id"]

            # Metadata
            fields = []
            fields.append(
                f"*<https://fink-portal.org/{cand['objectId']}|{cand['objectId']}>*\nFink: *{cand['classification']}*"
            )
            fields.append(f"*{time_str}*\nDetections: *{ndet}* Limits: *{nlim}*")
            fields.append(f"*RA/Dec*: {sc_equ}\n*Gal:* {sc_gal}")
            fields.append(
                f"*<{tns_link}|TNS>*: *{tns_info}*\n<{simbad_link}|SIMBAD> <{snad_link}|SNAD> <{aavso_link}|AAVSO>"
            )

            # Image thumbnail
            accessory = {
                "type": "image",
                "alt_text": "Pan-STARRS",
                "image_url": ps1_link,
            }

            result = slack_client.chat_postMessage(
                channel=slack_channel,
                text=cand["objectId"],
                blocks=[
                    {
                        "type": "section",
                        "fields": [{"type": "mrkdwn", "text": _} for _ in fields],
                        "accessory": accessory,
                    },
                    {
                        "type": "image",
                        "alt_text": cand["objectId"],
                        "slack_file": {"id": id_lc},
                    },
                    {"type": "divider"},
                ],
                unfurl_links=False,
            )
            slack_msgs.append(result)

            time.sleep(sleep)

        if send_to_tg:
            # Metadata
            fields = []
            fields.append(
                f"[*{cand['objectId']}*](https://fink-portal.org/{cand['objectId']}) : *{escape(cand['classification'])}*"
            )
            fields.append(
                f"*{escape(time_str)}*\nDetections: *{ndet}* Limits: *{nlim}*"
            )
            fields.append(f"*RA/Dec*: {escape(sc_equ)}\n*Gal:* {escape(sc_gal)}")
            fields.append(f"[TNS]({tns_link}): *{escape(tns_info)}*")
            fields.append(
                f"[SIMBAD]({simbad_link}) [SNAD]({snad_link}) [AAVSO]({aavso_link})"
            )

            candidate["lc"].seek(0)

            tg_msgs.append(["\n".join(fields), ps1_link, candidate["lc"]])

    if send_to_tg and tg_msgs:
        msg_handler_tg(
            tg_msgs,
            tg_channel,
            None,
            sleep_seconds=sleep,
            parse_mode="MarkdownV2",
            token=os.environ["EARLY_TDE_TG_TOKEN"],
        )

    # if send_to_slack:
    #     # Delete the Slack messages, useful for debugging
    #     time.sleep(10)

    #     for result in slack_msgs:
    #         slack_client.chat_delete(channel=result.data['channel'], ts=result.data['ts'])

    # TODO: propagate the scores to parent dataframe
    return candidates


if __name__ == "__main__":
    """Execute the test suite"""

    # Run the test suite
    globs = globals()
    spark_unit_tests(globs)
