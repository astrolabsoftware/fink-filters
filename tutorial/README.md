# Building a filter

## Before starting

In order to design a Fink filter, you will need to have Apache Spark installed (3.x):

```bash\
# Install Apache Spark
SPARK_VERSION=3.1.3
wget --quiet https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-${HADOOP_VERSION}.tgz
tar -xf spark-${SPARK_VERSION}-bin-${HADOOP_VERSION}.tgz
rm spark-${SPARK_VERSION}-bin-${HADOOP_VERSION}.tgz
```

and put these lines in your ~/.bash_profile:

```bash
export SPARK_HOME=/path/to/spark-${SPARK_VERSION}-bin-${HADOOP_VERSION}
export PATH=$PATH:$SPARK_HOME/bin
export PYTHONPATH=$PYTHONPATH:$SPARK_HOME/python
```

## Structure of a filter

Fink filters are contained in `fink_filters/`. They all have the same structure:

```
├── filter.py
├── filter_utils.py
└── __init__.py
```

- `__init__.py`: empty (module)
- `filter.py`: contains the Spark pandas UDF(s)
- `filter_utils.py`: Optional. Anything else you need to make your filter to work

## Testing your filter (tutorial)

Just launch:

```bash
PYSPARK_DRIVER_PYTHON=`which jupyter-notebook` `which pyspark`
```

You should have access to jupyter notebooks with Spark inside!

## What is next?

Once you are happy with your filter, open a PR and we will review it before merging it.
