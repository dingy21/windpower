# Wind Power Forecasting Project
Renewable energy remains one of the most important topics for a sustainable future. With the rise of wind farms, wind power forecasting has become an important topic. Utility companies need accurate forecasts of the amount of power being produced by wind turbines in order to ensure that they have sufficient power available to meet demand.
## Load Library
```
library(tidyverse)
library(tidymodels)
library(janitor)
library(skimr)
library(kableExtra)
library(GGally)
library(vip)        
library(fastshap)   
library(MASS)
library(ISLR)
library(tree)
library(ggplot2)
library(dplyr)
library(lubridate)
library(imputeTS)
library(imputeTS)
library(forecast)
library(urca)
library(pracma)
library(rlang)
library(astsa)
library(fpp2)
library(forecast)
```
## Import Data
```
Turbine <- read_csv("Turbine_Data.csv") %>% clean_names()
head(Turbine) 
```
## Data Preparation
```
# -- impute missing values with mean
Turbine$active_power[is.na(Turbine$active_power)] <- mean(Turbine$active_power, na.rm = T)
Turbine$ambient_temperature[is.na(Turbine$ambient_temperature)] <- mean(Turbine$ambient_temperature, na.rm = T)
Turbine$wind_direction[is.na(Turbine$wind_direction)] <- mean(Turbine$wind_direction, na.rm = T)
Turbine$wind_speed[is.na(Turbine$wind_speed)] <- mean(Turbine$wind_speed, na.rm = T)

# -- accumulate with a daily index
Turbine_day <- Turbine %>%
  filter(year > 2017) %>%   # filter out 2017 due to data quality issues
  group_by(year, month, day) %>%
  summarise(day_active_power = sum(active_power),
            ambient_temperature = mean(ambient_temperature),
            wind_direction = mean(wind_direction),
            wind_speed = mean(wind_speed)) %>%
  mutate(date = as.Date(paste(year, "-", month, "-", day), format = "%Y - %m - %d"))

head(Turbine_day,10)
```
## Exploratory Analysis
```
# -- find the target value pattern by time
Turbine_day %>%
  ggplot(aes(x = date, y = day_active_power)) +
  geom_line() +
  labs(title = "Daily Active Power by Date",
       y = "Active Power",
       x = "Date")

# -- plot relationship
ggplot(Turbine_day, aes(x=ambient_temperature, y=day_active_power)) + geom_point() 
ggplot(Turbine_day, aes(x=wind_direction, y=day_active_power)) + geom_point() 
ggplot(Turbine_day, aes(x=wind_speed, y=day_active_power)) + geom_point() 
```
## Create Time Series Object and Plot
```
turbine1 <- subset(Turbine_day, select=c(day_active_power))

Turbine_ts <- ts(turbine1,start = c(2018,1,1), frequency = 365)

plot(Turbine_ts)
ggAcf(Turbine_ts,lag.max=200)
ggPacf(Turbine_ts,lag.max=200)
```
## Test for White Noise
Ho: white noise
Ha: not white noise
```
Box.test(Turbine_ts, lag=8, fitdf=0, type="Lj")
```
## ADF Test for Stationarity
### Use Single Mean Version of the Test
Ho: non-stationary and need 1st difference
Ha: stationary
```
test1_df <- ur.df(Turbine_ts, type = "drift")
summary(test1_df)
```
## Model 1
### Which explanatory variables are important?
```
# -- use sarima to test for coefficient significance

fit_AR0 <- sarima(Turbine_ts, 0, 0, 0, xreg=Turbine_day[,5:7])
summary(fit_AR0)
fit_AR0

# -- use arima to forecast
fit_AR0 <- arima(Turbine_ts, order=c(0,0,0), xreg=Turbine_day[,5:7])
summary(fit_AR0)
fit_AR0

# -- examine the residuals
checkresiduals(fit_AR0)
```
## Model 1 Adjustment: excluding wind_direction
```
# -- remove insignificant variable wind direction
Turbine_day2 <- Turbine_day %>%
  subset(.,select=-c(wind_direction,day_active_power,year,month,day,date)) %>%
  as.matrix()

fit_AR0 <- sarima(Turbine_ts, 0, 0, 0, xreg=Turbine_day2[,1:2])
summary(fit_AR0)
fit_AR0

# -- using arima to get the forecast
fit_AR0 <- arima(Turbine_ts, order=c(0,0,0), xreg=Turbine_day2[,1:2])
summary(fit_AR0)
fit_AR0

# -- examine residuals of the model
checkresiduals(fit_AR0)
```
## Model 2: AR(1)
```
fit_AR1 <- sarima(Turbine_ts, 1, 0, 0, xreg=Turbine_day2[,1:2])
summary(fit_AR1)
fit_AR1

fit_AR1 <- arima(Turbine_ts, order=c(1,0,0), xreg=Turbine_day2[,1:2])
summary(fit_AR1)
fit_AR1

checkresiduals(fit_AR1)
```
## Model 3: ARIMA(1,0,1)
```
fit_AR101 <- sarima(Turbine_ts, 1, 0, 1, xreg=Turbine_day2[,1:2])
summary(fit_AR101)
fit_AR101

fit_AR101 <- arima(Turbine_ts, order=c(1,0,1), xreg=Turbine_day2[,1:2])
summary(fit_AR101)
fit_AR101

checkresiduals(fit_AR101)
```
## Model 4: auto.arima
```
fitauto <- auto.arima(Turbine_ts, xreg=as.matrix(Turbine_day2[,1:2]))
summary(fitauto)
checkresiduals(fitauto)
```
## Final Model: ARIMA(2,0,2)
```
fit_AR202 <- sarima(Turbine_ts, 2, 0, 2, xreg=Turbine_day2[,1:2])
summary(fit_AR202)
fit_AR202

fit_AR202 <- Arima(Turbine_ts, order=c(2,0,2), xreg=as.matrix(Turbine_day2[,1:2]))
summary(fit_AR202)
fit_AR202

checkresiduals(fit_AR202)
plot(residuals(fit_AR202))
```
## Make Prediction
```
# -- create a matrix of covariates for the next 5 time periods
new <- c(35.25,142.25,5.73,35.59,331.22,4.03,34.68,295.51,3.88,33.44,239.83,5.01,34.06,279.92,4.51)
new <- matrix(new, nrow = 5, ncol = 3, byrow = T)
xreg1 <- new[,-c(2)]

forecast(fit_AR202, xreg=xreg1, h=5)
autoplot(forecast(fit_AR202, xreg=xreg1, h=5))
```
