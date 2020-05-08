# Thomas Navidi
# Script for simulating mpc scheduler for battery only
# Fan is assumed to be full blast while temperature >= 72
# Now includes fan solar generation

import numpy as np
import cvxpy as cvx
# import pandas as pd

import matplotlib.pyplot as plt
import time

from statsmodels.tsa.statespace.sarimax import SARIMAX


class Forecaster:
    def __init__(self, my_order=(3, 0, 3), my_seasonal_order=(3, 0, 3, 24), pos=False, training_mean_p=None, training_mean_s=None):
        self.pos = pos  # boolean indicating whether or not the forecast should always be positive
        self.my_order = my_order
        self.my_seasonal_order = my_seasonal_order
        self.model_params_p = np.nan
        self.model_params_r = np.nan
        self.model_params_s = np.nan
        self.training_mean_p = training_mean_p
        self.training_mean_s = training_mean_s

    def input_training_mean(self, training_mean, model_name='p'):
        if model_name == 'p':
            self.training_mean_p = training_mean
        else:
            self.training_mean_s = training_mean
        return True

    def scenarioGen(self, pForecast, scens, battnodes):
        """
        Inputs: battnodes - nodes with storage
            pForecast - real power forecast for only storage nodes
            pMeans/Covs - dictionaries of real power mean vector and covariance matrices
                            keys are ''b'+node#' values are arrays
            scens - number of scenarios to generate
        Outputs: sScenarios - dictionary with keys scens and vals (nS X time)
        """

        nS, T = pForecast.shape
        sScenarios = {}
        for j in range(scens):
            counter = 0
            tmpArray = np.zeros((nS, T))
            if nS == 1:
                sScenarios[j] = pForecast  # no noise
            else:
                for i in battnodes:
                    # resi = np.random.multivariate_normal(self.pMeans['b'+str(i+1)],self.pCovs['b'+str(i+1)])
                    # tmpArray[counter,:] = pForecast[counter,:] + resi[0:T]
                    tmpArray[counter, :] = pForecast[counter, :]  # no noise
                    counter += 1
                sScenarios[j] = tmpArray

        return sScenarios

    def netPredict(self, prev_data_p, time):
        # just use random noise function predict
        pForecast = self.predict(prev_data_p, time, model_name='p')
        return pForecast

    def rPredict(self, prev_data_r, time):
        # just use random noise function predict
        rForecast = self.predict(prev_data_r, time, model_name='r')
        return rForecast

    def train(self, data, model_name='p'):
        model = SARIMAX(data, order=self.my_order, seasonal_order=self.my_seasonal_order, enforce_stationarity=False,
                        enforce_invertibility=False)
        if model_name == 'r':
            model_fitted_r = model.fit(disp=False)  # maxiter=50 as argument
            self.model_params_r = model_fitted_r.params
        elif model_name == 's':
            model_fitted_s = model.fit(disp=False)  # maxiter=50 as argument
            self.model_params_s = model_fitted_s.params
        else:
            model_fitted_p = model.fit(disp=False)  # maxiter=50 as argument
            self.model_params_p = model_fitted_p.params

    def saveModels(self, fname):
        np.savez(fname, model_fitted_p=self.model_fitted_p, model_fitted_r=self.model_fitted_r,
                 model_fitted_s=self.model_fitted_s)

    def loadModels(self, model_params_p=None, model_params_r=None, model_params_s=None):
        """
        self.model = SARIMAX(data, order=self.my_order, seasonal_order=self.my_seasonal_order,
                        enforce_stationarity=False, enforce_invertibility=False)
        self.model_fit = self.model.filter(model_fitted.params)
        """

        if model_params_p is not None:
            self.model_params_p = model_params_p
        if model_params_r is not None:
            self.model_params_r = model_params_r
        if model_params_s is not None:
            self.model_params_s = model_params_s

    def predict(self, prev_data, period, model_name='p'):

        # stime = time.time()

        model = SARIMAX(prev_data, order=self.my_order, seasonal_order=self.my_seasonal_order,
                        enforce_stationarity=False, enforce_invertibility=False)
        if model_name == 'r':
            model_fit = model.filter(self.model_params_r)
        elif model_name == 's':
            model_fit = model.filter(self.model_params_s)
        else:
            model_fit = model.filter(self.model_params_p)

        yhat = model_fit.forecast(period)

        if self.pos:
            yhat = yhat.clip(min=0)  # do not allow it to predict negative values for demand or solar

        # print 'pred time', time.time()-stime

        return yhat


class Resampling:  # resamples data to proper time resolution
    def __init__(self, data, t, tnew):
        self.data = data
        self.t = t
        self.tnew = tnew

    def downsampling(self, data, t, tnew):
        t_ratio = tnew // t
        downsampled = np.zeros((np.size(data, 0), np.size(data, 1) / t_ratio))
        for i in range(np.size(downsampled, 0)):
            for j in range(np.size(downsampled, 1)):
                downsampled[i][j] = np.average(data[i][j * t_ratio:(j + 1) * t_ratio])
        return downsampled

    def upsampling(self, data, t, tnew, new_num_col):
        steps_per_day = int(1440 / t)
        # num_days = int((np.size(data) * t) / 1440)
        one_day = np.zeros((np.size(data, 0), steps_per_day))
        one_day_std = np.zeros((np.size(data, 0), steps_per_day))

        for a in range(np.size(one_day, 0)):
            for b in range(np.size(one_day, 1)):
                sub_array = data[a, b::steps_per_day]
                one_day[a, b] = np.mean(sub_array)
                one_day_std[a, b] = np.std(sub_array)

        t_ratio = t // tnew
        upsampled = np.zeros((np.size(data, 0), new_num_col))
        for i in range(np.size(upsampled, 0)):
            for j in range(np.size(upsampled, 1)):
                # upsampled[i][j] = np.random.normal(one_day[i,(j/t_ratio)%24], one_day_std[i,(j/t_ratio)%24])
                upsampled[i][j] = np.random.normal(data[i, (j / t_ratio)], one_day_std[i, (j / t_ratio) % 24])
        return upsampled


class Controller(object):

    def __init__(self, d_price, b_price, h_price, Qmin, Qmax, cmax, dmax, fmax, hmax, l_eff, c_eff, d_eff, coeff_f,
                 coeff_x, coeff_h, coeff_b, n_f, n_t, n_b, T, t_res=15. / 60):
        # coeffs are fan model coefficients
        # self.prices = prices entered when solving
        self.d_price = d_price
        self.b_price = b_price
        self.h_price = h_price  # cost of deviating above THI
        self.Qmin = Qmin
        self.Qmax = Qmax
        self.cmax = cmax
        self.dmax = dmax
        self.fmax = fmax
        self.hmax = hmax
        self.l_eff = l_eff
        self.c_eff = c_eff
        self.d_eff = d_eff
        self.t_res = t_res  # data time resolution in hours
        self.coeff_f = coeff_f
        self.coeff_x = coeff_x
        self.coeff_h = coeff_h
        self.coeff_b = coeff_b.flatten()
        self.n_f = n_f  # number of fans
        self.n_t = n_t  # number of temperature sensors
        self.n_b = n_b  # number of batteries
        self.T = T  # length of optimization horizon

    def fanPrediction(self, h0, f_exo):
        # f_exo are the fan exogenous inputs for the fan model, such as outdoor temperature
        # make fan prediction

        N, T = f_exo.shape
        h = np.zeros((self.n_t, T + 1))
        f_p = np.zeros((self.n_f, T))

        h[:, 0] = h0.flatten()

        for t in range(T):
            if np.max(h[:,t]) >= self.hmax:
                f_p[:, t] = self.fmax * np.ones(self.n_f)
            else:
                f_p[:, t] = np.zeros(self.n_f)

            h[:, t + 1] = np.dot(self.coeff_h, h[:, t]) + np.dot(self.coeff_x, f_exo[:, t]) \
                          + np.dot(self.coeff_f, f_p[:, t]) + self.coeff_b

        return f_p, h

    def fanPredictionSimple(self, f_on, time_curr, f_start=8*4, f_end=21*4):
        # f_on is binary indicating if the fans are currently on or not
        # time_curr is the number of 15 minute time steps past midnight it is now
        # default assume the fans start at 8am and end at 9pm

        f_p = np.zeros((self.n_f, self.T))
        if f_on and time_curr < f_end:
            # fan is currently running -> predict will run until end
            duration = f_end - time_curr
            fan = self.fmax * np.ones((self.n_f, duration))
            start = 0
        elif f_on and time_curr >= f_end:
            # fan is on after expected time -> fan will remain on one more period
            duration = 1
            fan = self.fmax * np.ones((self.n_f, duration))
            start = 0
        elif time_curr <= f_start:
            # no fan before start -> predict fan on at start until end
            duration = f_end - f_start
            fan = self.fmax * np.ones((self.n_f, duration))
            start = f_start - time_curr
        elif time_curr >= f_start and time_curr < f_end:
            # no fan after expected time -> predict fan will run next period
            duration = f_end - time_curr - 1
            fan = self.fmax * np.ones((self.n_f, duration))
            start = 1
        elif time_curr >= f_end:
            # no fan after end time -> fan will stay off until tomorrow
            start = self.T - time_curr + f_start
            duration = self.T - start
            if duration > f_end - f_start:
                duration = f_end - f_start
            fan = self.fmax * np.ones((self.n_f, duration))
            start = self.T - time_curr + f_start

        f_p[:, start:start + duration] = fan

        return f_p

    def optimize(self, power, solar, prices, Q0, Pmax0, f_p):
        # f_p is predicted fan power consumption
        n, T = power.shape

        cmax = np.tile(self.cmax, (self.n_b, T))
        dmax = np.tile(self.dmax, (self.n_b, T))
        Qmin = np.tile(self.Qmin, (self.n_b, T + 1))
        Qmax = np.tile(self.Qmax, (self.n_b, T + 1))
        #fmax = np.tile(self.fmax, (self.n_f, T))
        #hmax = np.tile(self.hmax, (self.n_t, T + 1))
        solar = np.tile(solar, (self.n_f, 1))

        # print(solar.shape)
        # print(solar)

        c = cvx.Variable((self.n_b, T))
        d = cvx.Variable((self.n_b, T))
        #f = cvx.Variable((self.n_f, T))
        Q = cvx.Variable((self.n_b, T + 1))
        #h = cvx.Variable((self.n_t, T + 1))

        # Battery, fan, THI, Constraints
        constraints = [c <= cmax,
                       c >= 0,
                       d <= dmax,
                       d >= 0,
                       #f >= 0,
                       #f <= fmax,
                       Q[:, 0] == Q0,
                       Q[:, 1:T + 1] == self.l_eff * Q[:, 0:T]
                       + self.c_eff * c * self.t_res - self.d_eff * d * self.t_res,
                       Q >= Qmin,
                       Q <= Qmax,
                       #h[:, 0] == h0.flatten()
                       ]

        # THI vs fan power model
        #for t in range(0, T):
        #    constraints.append(
        #        h[:, t + 1] == self.coeff_h * h[:, t] + np.dot(self.coeff_x, f_exo[:, t])
        #        + self.coeff_f * f[:, t] + self.coeff_b)

        # not a constraint, just a definition
        net = cvx.hstack([power.reshape((1, power.size)) + c - d
                          + cvx.reshape(cvx.sum(cvx.pos(f_p - solar), axis=0), (1, T)) - Pmax0, np.zeros((1, 1))])

        obj = cvx.Minimize(
            cvx.sum(cvx.multiply(prices, cvx.pos(power.reshape((1, power.size))
                                                 + c - d + cvx.reshape(cvx.sum(cvx.pos(f_p - solar), axis=0),
                                                                       (1, T)))))  # cost min
            + self.d_price * cvx.max(net)  # demand charge
            + self.b_price * cvx.sum(c + d)  # battery degradation
            # attempt to make battery charge at night * doesnt do anything
            # + 0.0001 * cvx.sum_squares((power.reshape((1, power.size))
            #                            + c - d + cvx.reshape(cvx.sum(cvx.pos(f_p - solar), axis=0), (1, T)))/np.max(power))
        #   + self.h_price * cvx.sum_squares(cvx.pos(h - hmax))  # THI penalty
        )

        prob = cvx.Problem(obj, constraints)

        prob.solve(solver=cvx.ECOS)

        # calculate expected max power
        net = power + c.value - d.value + np.sum(np.clip(f_p, 0, None), axis=0) - Pmax0
        Pmax_new = np.max(net) + Pmax0
        if Pmax_new < Pmax0:
            Pmax_new = Pmax0

        return c.value, d.value, Q.value, prob.status, Pmax_new


def trainForecaster(data_p, n_samples, fname):
    # train models

    data_p = data_p.reshape((data_p.size, 1))

    # for training
    training_data = data_p[0:n_samples]

    # print('training data', training_data)

    training_mean = np.mean(training_data)
    print('mean', training_mean)

    forecaster = Forecaster(my_order=(3, 0, 3), my_seasonal_order=(3, 0, 3, 24), pos=False)
    forecaster.train(training_data, model_name='p')
    forecaster_params = forecaster.model_params_p

    np.savez('forecast_models/' + fname + str(n_samples) + '.npz', forecaster_params=forecaster_params,
             training_mean=training_mean)
    print('SAVED forecaster params at', 'forecast_models/' + fname + str(n_samples) + '.npz')

    return forecaster, training_mean


def testForecaster(data_p, n_samples):
    # test
    # for loading the forecaster from a saved file
    model_data = np.load('SARIMA_model_params' + str(n_samples) + '.npz')
    forecaster_params = model_data['forecaster_params']
    training_mean = model_data['training_mean']

    forecaster = Forecaster(my_order=(3, 0, 3), my_seasonal_order=(3, 0, 3, 24), pos=False)
    model_fitted_p = forecaster_params
    forecaster.loadModels(model_params_p=model_fitted_p)

    # begin test
    yhats = []
    offset = 4 * 3 * 24  # amount of data points needed for forecaster predictions
    iters = 4 * 24 * 3
    period = 4 * 24
    step = 1  # test error of only 1 step ahead prediction
    for i in range(iters):
        prev_data_p = data_p[n_samples - offset + step * i:n_samples + step * i]
        prev_data_p = prev_data_p.reshape((prev_data_p.size, 1))
        yhat = forecaster.predict(prev_data_p, period, model_name='p')

        # yhat += -np.min(yhat)
        yhat += training_mean

        yhats.append(yhat[0:step])  # only save next step

    yhat = np.concatenate(yhats)
    y_real = data_p[n_samples:n_samples + step * (i + 1)]

    print('MAE', np.mean(np.abs(y_real - yhat)))
    print('mean', np.mean(y_real))

    """
    plt.figure()
    plt.plot(yhat)
    plt.plot(y_real)
    plt.show()
    """

    return forecaster


def preparePowerForecaster(n_samples, training_data=None):
    # SARIMA forecaster for power
    if training_data is not None:
        # train SARIMA forecaster
        print('no forecaster data found, so training new forecaster')
        forecaster, training_mean = trainForecaster(training_data, n_samples, 'SARIMA_model_params')
        forecaster.input_training_mean(training_mean, model_name='p')
    else:
        # load SARIMA forecaster
        print('Loading forecaster parameters from', 'forecast_models/SARIMA_model_params' + str(n_samples) + '.npz')
        model_data = np.load('forecast_models/SARIMA_model_params' + str(n_samples) + '.npz')
        forecaster_params = model_data['forecaster_params']
        training_mean = model_data['training_mean']
        forecaster = Forecaster(my_order=(3, 0, 3), my_seasonal_order=(3, 0, 3, 24), pos=False)
        model_fitted_p = forecaster_params
        forecaster.loadModels(model_params_p=model_fitted_p)
        forecaster.input_training_mean(training_mean, model_name='p')

    return forecaster


def prepareSolarForecaster(n_samples, training_data=None):
    if training_data is not None:
        # train SARIMA forecaster
        print('no solar forecaster data found, so training new forecaster')
        forecaster_s, training_mean_s = trainForecaster(training_data, n_samples, 'SARIMA_SolarModel_params')
        forecaster_s.input_training_mean(training_mean_s, model_name='s')
    else:
        # load SARIMA forecaster
        print('Loading solar forecaster parameters from',
              'forecast_models/SARIMA_SolarModel_params' + str(n_samples) + '.npz')
        model_data = np.load('forecast_models/SARIMA_SolarModel_params' + str(n_samples) + '.npz')
        forecaster_params = model_data['forecaster_params']
        training_mean_s = model_data['training_mean']
        forecaster_s = Forecaster(my_order=(3, 0, 3), my_seasonal_order=(3, 0, 3, 24), pos=False)
        model_fitted_p = forecaster_params
        forecaster_s.loadModels(model_params_p=model_fitted_p)
        forecaster_s.input_training_mean(training_mean_s, model_name='s')

    return forecaster_s



def realTimeUpdate(p_net, pmax, u_curr, price_curr):
    # takes the most current readings and makes real time adjustments
    # purpose is to compensate for forecaster errors

    if p_net > pmax:
        u_new = u_curr + pmax - p_net
    elif p_net < 0:
        u_new = u_curr - p_net
    else:
        u_new = u_curr

    return u_new


def simulateFans(time_curr, period, f_on=0):
    # randomly generate fan on signal (should be real input for real implementation)
    start_time = np.random.choice([7, 8, 9, 10], 1)
    end_time = np.random.choice([20, 21, 22, 23], 1)
    if f_on and time_curr < end_time * 4:
        f_on = 1
    elif f_on and time_curr > end_time * 4:
        f_on = 0
    elif time_curr >= start_time * 4 and time_curr < period / 2:
        f_on = 1
    else:
        f_on = 0

    if f_on == 1:  # add random chance for fan to turn off when it should be on (simulate cows leaving barn)
        if np.random.uniform(0, 1) < 1. / (12 * 4):
            f_on = 0

    return f_on


def initSimulationData(power_size, n_f, n_t):
    c_all = np.zeros(power_size)
    d_all = np.zeros(power_size)
    f_all = np.zeros((n_f, power_size))
    h_all = np.zeros((n_t, power_size))
    Q_all = np.ones(power_size)
    net_all = np.zeros(power_size)

    return c_all, d_all, f_all, h_all, Q_all, net_all

def getHistoryforForecast(power, start_idx, offset, t_horizon, i):
    prev_data_p = power[:, start_idx - offset + t_horizon * i:start_idx + t_horizon * i]
    prev_data_p = prev_data_p.reshape((prev_data_p.size, 1))
    return prev_data_p

def staticData(n_f):
    # This data should be kept the same

    h_price = None
    b_price = 0.001  # make price of battery degradation low for now

    t_res = 15 / 60.  # 15 minutes is 15/60 of an hour

    period = 4 * 24  # number of time steps in optimization
    offset = 4 * 24 * 3  # amount of data points needed for forecaster predictions

    t_horizon = 1  # 4*12 # how many 15 minute periods between each run (should be 1 in practice)
    t_lookahead = 24 * 4  # how many 15 minute periods to look ahead in optimization

    hmax = 1

    # leakage < 1 encourages the optimization to charge immediately before it is needed which can be risky
    # l_eff = 0.9995  # 15 minute battery leakage efficiency = 94.4% daily leakage
    l_eff = 1.0001
    c_eff = 0.975  # charging efficiency
    d_eff = 1 / 0.975  # round trip efficiency = 0.95

    n_t = 8
    n_b = 1
    coeff_f = -np.eye(n_f)
    coeff_x = 0.2 * 3.4 * np.eye(n_f)  # not necessarily square
    coeff_h = np.eye(n_t)  # not necessarily square
    coeff_b = np.zeros((n_f, 1))

    n_samples = 4 * 24 * 7 # 1 week of training data for forecasters

    return h_price, b_price, t_res, period, offset, hmax, l_eff, c_eff, d_eff, n_t, n_b, \
           coeff_f, coeff_x, coeff_h, coeff_b, n_samples, t_horizon, t_lookahead


def adjustedStaticData(t_res=15. / 60.):
    ### This function sets prices (TOU, Demand Charge), battery and fan param
    # this data should be adjusted once before running the first time

    # TOU charge
    s_peak = 0.31752        # $/kWh
    s_offpeak = 0.16888     # $/kWh
    prices = np.hstack((s_offpeak * np.ones((1, 12 * 4)), s_peak * np.ones((1, 6 * 4)), s_offpeak * np.ones((1, 6 * 4)))) / 4 / 2
    prices_full = np.reshape(np.tile(prices, (1, 31)), (31 * 24 * 4, 1)).T
    prices_full = prices_full * t_res  # scale prices to reflect hours instead of true time resolution

    # demand charge
    # d_price = 18.26

    # battery info
    # Qmax = 17 * 24 * 0.2
    # print('Battery energy capacity (kWh)', Qmax)
    # cmax = Qmax / 3.  # max charging rate
    # print('Battery power capacity (kW)', cmax)
    # dmax = cmax  # max discharging rate
    # fmax = 1. / 8. * cmax  # max power consumed by a single fan
    # print('maximum fan power for a single fan', fmax)
    # print('shape of uncontrollable demand', power.shape)

    # n_f = 8 # the number of fans

    d_price = 10.77     # demand charge $/kW

    # batt info
    Qmin = 0            # min battery energy
    Qmax = 10           # max battery energy
    cmax = 8            # max battery charging rate kW
    dmax = 8            # max battery discharging rate kW

    # fan info
    fmax = 1.7  # max power consumed by a single fan kW
    n_f = 15 # number of fans


    return prices_full, d_price, Qmin, Qmax, cmax, dmax, fmax, n_f


def dynamicData(d_name, s_name):
    # this data should changed and input each time

    # load simulation power data
    power = np.loadtxt(d_name)
    power = power.reshape((1, len(power)))

    # load simulation solar data
    solar_full = np.loadtxt(s_name)

    solar_full = solar_full.reshape((1, len(solar_full))).clip(min=0)  # remove negative entries

    # s_name will be solar_data
    # solar_full = solar_data.to_numpy()

    # ~~~~~~~~~~~~~~ REMOVE THIS LINE FOR REAL IMPLEMENTATION
    solar_full = solar_full * 3.4

    # Where in the dataset are we starting the simulation
    # for a standard run, the dataset should consist of at least 3 days of historical data (4 * 24 * 3 points)
    # This is so the forecaster can use the historical data to make a prediction
    # when training the forecaster, the dataset should be at least 7 days of history so start_idx = 4 * 24 * 7 points
    start_idx = 4 * 24 * 3

    # keep track of night to improve solar forecast. (ex. night time from 6am to 8pm in July)
    # data can be found at https://www.timeanddate.com/sun/usa/palo-alto
    sun_start = 6 * 4  # number of 15 minute intervals after midnight that the sun rises
    sun_stop = 20 * 4
    # creating the solar mask
    night_mask = np.hstack((np.zeros(sun_start, dtype=int), np.ones(sun_stop - sun_start, dtype=int),
                            np.zeros(4 * 4, dtype=int)))
    night_mask_full = np.tile(night_mask, 31)

    # approximate start and end times of the fans to improve predictions
    f_start = 7 * 4
    f_end = 20 * 4

    f_on = 0

    # initial battery charge in kWh
    Q0 = 0.1

    # last peak power consumption in billing period
    Pmax0 = 1.5

    # current time (number of 15 minute intervals past midnight)
    time_curr = 0

    return power, solar_full, start_idx, f_start, f_end, f_on, Q0, night_mask_full, Pmax0, time_curr


def main(power, solar_full, start_idx, f_start, f_end, f_on, Q0, night_mask_full, Pmax0, time_curr):
    # Dynamic data is input into main function

    # Define all the needed static info
    ### This function sets the prices
    prices_full, d_price, Qmin, Qmax, cmax, dmax, fmax, n_f = adjustedStaticData() ############# DONE ###########

    ### Change n_f to the real number of fans = 15
    h_price, b_price, t_res, period, offset, hmax, l_eff, c_eff, d_eff, n_t, n_b, \
        coeff_f, coeff_x, coeff_h, coeff_b, n_samples, t_horizon, t_lookahead = staticData(n_f) ############# DONE ###########

    # initialize controller
    contr = Controller(d_price, b_price, h_price, Qmin, Qmax, cmax, dmax, fmax, hmax, l_eff, c_eff, d_eff, coeff_f,
                       coeff_x, coeff_h, coeff_b, n_f, n_t, n_b, period, t_res) ############# DONE ###########

    # ~~~~~~~ Uncomment these when training the forecaster
    # for training the forecaster input at least 1 week of historical data
    # if it gives a ConvergenceWarning when training, just ignore it.
    # forecaster = preparePowerForecaster(n_samples, training_data=power)
    # forecaster_s = prepareSolarForecaster(n_samples, training_data=solar_full)

    # ~~~~~~~~~~~ Uncomment these when running the forecaster without training
    forecaster = preparePowerForecaster(n_samples)
    forecaster_s = prepareSolarForecaster(n_samples)

    # c_all, d_all, f_all, h_all, Q_all, net_all = initSimulationData(power.size, n_f)
    ############# DONE ###########

    # Run MPC control for 1 iteration

    # Get total barn power forecast
    prev_data_p = getHistoryforForecast(power, start_idx, offset, t_horizon, 0)
    p_curr = forecaster.predict(prev_data_p, period, model_name='p')
    p_curr = p_curr.reshape((1, p_curr.size)) + forecaster.training_mean_p

    # get solar power forecast ############# DONE ###########
    prev_data_s = getHistoryforForecast(solar_full, start_idx, offset, t_horizon, 0)
    solar_curr = forecaster_s.predict(prev_data_s, period, model_name='p')  # no difference in model name
    solar_curr = solar_curr.reshape((1, solar_curr.size)) + forecaster_s.training_mean_s
    night_mask = night_mask_full[time_curr:time_curr + t_lookahead]
    solar_curr = solar_curr * night_mask

    p_curr = np.zeros(solar_curr.size)
    p_curr = p_curr.reshape(1, p_curr.size)

    # get prices for current time ############# DONE ###########
    prices_curr = prices_full[:, time_curr:time_curr + t_lookahead]

    # simplified fan prediction that only uses current time and indicator if fan is on

    # ------ not needed in real run ------
    # randomly generate fan on signal (should be real input for real implementation)
    # f_on = simulateFans(time_curr, period)
    # ----- end not needed -------

    f_p = contr.fanPredictionSimple(f_on, time_curr, f_start, f_end)

    c_s, d_s, Q_s, prob_s, Pmax_ex = contr.optimize(p_curr, solar_curr, prices_curr, Q0, Pmax0, f_p)

    # you can use these plots to help you see what the optimization is doing

    plt.figure()
    plt.plot((p_curr + n_f * (f_p - solar_curr)).T)
    plt.plot((c_s - d_s).T)
    plt.plot((p_curr + c_s - d_s + n_f * (f_p - solar_curr)).T)
    plt.legend(('power, fan, solar', 'battery', 'net all'))
    plt.figure()
    plt.plot(p_curr.T)
    plt.plot(n_f * (f_p-solar_curr).T)
    plt.legend(('power', 'fan with solar'))
    plt.show()



    if prob_s != 'optimal':
        print('optimization status', prob_s)

    print('Old max power', Pmax0, 'new expected maximum power', Pmax_ex)

    # ----- not needed in real run -----
    # p_real = power[:, start_idx + i * t_horizon:start_idx + i * t_horizon + t_lookahead]
    # s_real = solar_full[:, start_idx + i * t_horizon:start_idx + i * t_horizon + t_lookahead]

    # real time demand charge and over-discharge correction
    # has minor improvement when solar forecast is wildly off
    # Can try just running the full algorithm again since run time is only 3 seconds
    # p_net = p_real + c_s - d_s + np.sum(np.clip(f_s - s_real, 0, None), axis=0)
    # ------- end not needed -----

    # input real time values for net power, battery power (u_curr = charge - discharge power), price
    real_time_data_point_for_net_power = 10
    p_net = real_time_data_point_for_net_power
    u_curr = c_s[:, 0] - d_s[:, 0]
    price_curr = prices_curr[:, 0]

    u_new = realTimeUpdate(p_net, Pmax_ex, u_curr, price_curr)
    if u_new != u_curr:
        print('value of u changed from', u_curr)
        print('to', u_new)
        if u_new < 0:
            d_s[0] = -u_new
            c_s[0] = 0
        else:
            c_s[0] = u_new
            d_s[0] = 0

    # c_s[0] and d_s[0] are the updated charge and discharge power level

    return c_s[0], d_s[0]


if __name__ == '__main__':
    # MPC optimization is run every 15 minutes
    # time resolution of data is 15 minutes

    # forecast training should be at least 1 week of historical data
    # normal runs on a pre-trained forecaster should have 3 days of historical data
    st = time.time()

    # loading the dynamic data to be input to the main function
    power, solar_full, start_idx, f_start, f_end, f_on, Q0, night_mask_full, Pmax0, time_curr = \
        dynamicData('synthFarmData_15minJuly.csv', 'synthSolarData_15minJuly.csv')

    c, d = main(power, solar_full, start_idx, f_start, f_end, f_on, Q0, night_mask_full, Pmax0, time_curr)

    print('run time', time.time() - st)
