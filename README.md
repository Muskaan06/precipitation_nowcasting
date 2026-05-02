---
license: cc-by-4.0
tags:
- weathernowcasting
- computervision
- earthscience
- dataset
pretty_name: SONIC The Dataset for weather nowcasting
size_categories:
- 10K<n<100K
---
# Dataset Card for SONIC

Satellite Observations Nowcasting for Indian subContinent or SONIC is the first sattelite-only rainfall nowcasting dataset spanning over the Indian subcontinent (4 major regions).

## Dataset Details
```
Regions Covered - Northern India(Jammu and Kashmir, `NI` folder), Southern India(Tamil nadu, Kerela, `SI` folder), North-Eastern India(7 Sisters, `7S` folder) and Easter India(Jharkhand, Odissa and West Bengal `JOB` folder)
```
```
Spatial Resolution - 8km/pixel
Spatial Coverage - 896x896 km
```
```
Temporal Resolution - 30 mins
Temporal Coverage - 2021-2024
```

## Dataset Structure

4 folders, each based on a different region of the Indian Subcontinent.
```
NI - 1565 events 
7S - 3885 events
JOB - 4250 events
SI - 3851 events
```
An event is a set of 24 sattelite images over the region with rainfall selected on the basis of a statistical algorithm.

Each image has 3 channels:-

```
Channel 0 - Rain (Measured in mm/hr)
Channel 1 - Latitude (Degrees)
Channel 2 - Longitude(Degrees)
```
