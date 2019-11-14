# Fink filters

This repository contains filters used to define which information will be sent to the community. 

## Step 0: Fork this repository

For the repository, and create a new folder at the root of the repo. The name of the repo does not matter much, but try to make it meaningful as much as possible! Let's call it `filter_rrlyr` for the sake of this example.

## Step 1: Define your filter

A filter is typically a Python routine that selects which alerts need to be sent based on user-defined criteria. Criteria are based on the alert entries: position, flux, properties, ... You can find what's in alert here [link to be added]. 

In this example, let's imagine you want to receive all alerts flagged as RRLyr by the xmatch module. you would define a simple routine 

```python
@pandas_udf(BooleanType(), PandasUDFType.SCALAR) # <- mandatory
def rrlyr(cross_match_alerts_per_batch: Any) -> pd.Series:
    """ Return alerts identified as RRLyr by the xmatch module.

    Parameters
    ----------
    cross_match_alerts_per_batch: Spark DataFrame Column
        Alert column containing the cross-match values

    Returns
    ----------
    out: pandas.Series of bool
        Return a Pandas DataFrame with the appropriate flag: 
        false for bad alert, and true for good alert.

    """
    mask = cross_match_alerts_per_batch.values == "RRLyr"

    return pd.Series(mask)
```

Remarks:

- Note the use of the decorator is mandotory. It is a Spark decorator, and specifies the output type, as well as the type of operation. Just copy and paste it.
- The name of the routine will be used as the name of the Kafka topic. So once the filter loaded, you would subscribe to the topic `rrlyr` to receive alerts from this filter. Hence choose a meaningful name!
- The name of the input argument must match the name of an alert entry. Here `cross_match_alerts_per_batch` is one column added by the xmatch module.
- You can have several input columns. Just add them one after the other:


```python
@pandas_udf(BooleanType(), PandasUDFType.SCALAR) # <- mandatory
def filter_w_several_input(acol: Any, anothercol: Any) -> pd.Series:
    """ Documentation """
    pass
```

## Step 3: Open a pull request

Once your filter is done, we will review it. The criteria for acceptance are:

- The filter works ;-)
- The volume of data to be transfered is tractable on our side. Keep in mind, LSST incoming stream is 10 million alerts per night, or ~1TB/night. Hence your filter must focus on a specific aspect of the stream, to reduce the outgoing volume of alerts. Based on your submission, we will provide estimate of the volume to be transfered.

## Step 4: Play!

If your filter is accepted, it will be plugged in the broker, and you will be able to receive your alerts in real-time using the [fink-client](https://github.com/astrolabsoftware/fink-client). Note that we do not keep alerts forever available in the broker. While the retention period is not yet defined, you can expect emitted alerts to be available no longer than one week.