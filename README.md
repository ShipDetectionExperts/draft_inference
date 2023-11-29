# Vessel detection as an Application Package

[![Docker Image Version (latest semver)](https://img.shields.io/docker/v/potato55/ship-detection?sort=semver)](https://hub.docker.com/r/potato55/ship-detection/)
[![Docker Pulls](https://img.shields.io/docker/pulls/potato55/ship-detection)](https://hub.docker.com/r/potato55/ship-detection/)

This repository contains a **work in progress** attempt to create an Application Package for vessel detection. The Application Package is intended to run on the EOEPCA platform.

## What is an Application Package?

A platform independent and self-contained representation of an Application, providing executables, metadata and dependencies such that it can be deployed to and executed within an Exploitation Platform. Learn more about the Application Package [here](https://docs.ogc.org/bp/20-089r1.html#toc0).

## What is the vessel detection application package?

The vessel detection application package is an ongoing project of demonstrating the capabilities of the EOEPCA platform. The Application Package contains a machine learning model trained to detect vessels on Sentinel 2 imagery.

## How to use the vessel detection application package?

There are two main ways of running an application package: locally and on the EOEPCA platform.

### Local execution

[cwltool](https://github.com/common-workflow-language/cwltool) can be used to execute the application package locally. The following command can be used to execute the application package locally. For example:

```bash
cwltool ship-detection.cwl#ship-detection-workflow --client_id "INSERT ID HERE" --client_secret "INSERT SECRET HERE" --bbox "13.4,52.5,13.6,52.7"  --time "2021-05/2021-08" --maxcc 20 --threshold 0.2
```

**--client_id** and **--client_secret** are the credentials for the Sentinel Hub API. You can get them by registering [here](https://www.sentinel-hub.com/).

**--bbox** is the bounding box of the area of interest. It is a comma separated list of coordinates in the following order: minx, miny, maxx, maxy.

**--time** is the time interval of the area of interest. It is a comma separated list of dates in the following order: start date, end date. It is important to note that the sentinel hub API only accepts time intervals, however the inference only runs on one image.

**--maxcc** is the maximum cloud coverage of the area of interest. It is a number between 0 and 100.

**--threshold** is the threshold for the inference. It is a number between 0 and 1.

### EOEPCA platform

Please see sample_requests.http for an example of how to use the application package on the EOEPCA platform. It is a set of preconfigured requests for the [Visual Studio Code REST Client extension](https://marketplace.visualstudio.com/items?itemName=humao.rest-client). The requests can be executed by clicking on the "Send Request" button. The variables are configured for the EOEPCA deployment guide setup, which is ideal for hosting your own instance of the EOEPCA platform and test applications. Please follow the [Deployment Guide](https://deployment-guide.docs.eoepca.org/current/) for more information.
