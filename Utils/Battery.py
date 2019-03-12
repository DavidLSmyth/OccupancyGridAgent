# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 16:08:00 2019

@author: 13383861
"""

import typing
from scipy.interpolate import splrep, splev
import matplotlib.pyplot as plt
#%%

#################### Interpolated Battery Model ####################
#In PX4 on Gazebo, ROS and other simulation softwares, there are usually simple battery models (there are many variables
#like wind and battery age that are difficult to calibrate which make the battery performance hard to model). For now, will assume
#that the agent has an effective range at a given speed, e.g. at 1m/s it can travel 3000m but at 2m/s it can only travel 1000m.
#These data points will be interpolated to give the full battery model.
#The agent will be configured with a battery, which will have an effective range at a given speed (e.g. 3000m at 2m/s).
#Given the spline model f (which maps speed to effective range at that speed), if the agent then travels 40m at 3m/s, 
#the battery capacity is set to (f(speed) - actual distance travelled) / f(speed)


#https://www.omnicalculator.com/other/drone-flight-time#drone-flight-time-formula

#flight time is usually given by time = capacity * discharge / AAD, where
#
#Time is the flight time of the drone, expressed in hours.
#Capacity is the capacity of your battery, expressed in milliamp hours (mAh) or amp hours (Ah). You can find this value printed on your LiPo battery. 
#The higher the capacity, the more energy is stored in the battery.

#Discharge is the battery discharge that you allow for during the flight. As LiPo batteries can be damaged if fully discharged, 
#it's common practice never to discharge them by more than 80%. If you'd like to change this default value, type the required discharge 
#into the respective field of this drone flight time calculator.

#AAD is the average amp draw of your drone, calculated in amperes. If you know this value, open the advanced mode to enter it directly 
#into our calculator. If you're not sure how to calculate it, keep reading - we will help you determine the amp draw basing on parameters 
#such as the quadcopter weight or battery voltage.

#Average amp draw can be calculated as follows: AAD = AUW * P / V
#AAD is the average amp draw, expressed in amperes;
#
#AUW is the all up weight of your drone - the total weight of the equipment that goes up in the air, including the battery. It is usually measured in kilograms.
#
#P is the power required to lift one kilogram of equipment, expressed in watts per kilogram. Our drone flight time 
#calculator assumes a conservative estimate of 170 W/kg. Some more efficient systems can take less, for example, 120 W/kg; 
#if that's your case, don't hesitate to adjust its value.
#
#V is the battery voltage, expressed in volts. You will find this value printed on your battery.
#

#If you're familiar with the Ohm's law, you probably noticed that P / V is the definition of an electric current I. Hence, you can use an alternative version of the formula above:
##AAD = AUW * I
##where I stands for the current (in amps) required to lift one kilogram into the air.


#for now make a conservative guess at all the parameters. Eventually move these to a config file
#80% maximum discharge, 20% typically always left in reserve (otherwise the battery gets damaged)
maximum_discharge = 0.8
#capacity measured in amp hours
battery_capacity = 8.8
battery_voltage = 36
#assume a weight of 2.5kg with the battery
agent_weight = 2.5
#170W/kg power to weight, i.e. how much power needs to lift one kg 
power_to_weight = 170
average_amp_draw = agent_weight * power_to_weight / battery_voltage
#flight time calculated in hours
flight_time = battery_capacity * maximum_discharge / average_amp_draw
#this is the RAVs effective flight time in seconds
flight_time_seconds = flight_time*60*60

#For another time: figure out from above how to generate a discharge model given conditions


#https://airdata.com/blog/2017/drone-flight-stats-part-1
#take graph 2 here and interpolate
speeds_ranges = [_ for _ in range(0, 41, 5)]
flight_times = [6.3, 6.3, 6.15, 5.85, 5.3, 4.85, 4.6, 4.1, 3.6]
effective_capacities = [flight_time / max(flight_times) for flight_time in flight_times]
#effective capacity given that rav is flying at a certain speed
effective_capacity_given_speed_spline = splrep(speeds_ranges, effective_capacities)


plt.plot([i for i in range(40)], [splev(i, effective_capacity_given_speed_spline ) for i in range(40)])
#################### Interpolated Battery Model ####################


#%%
class RAVBattery:
    
    '''
    A battery class to be used with a Remote Aerial Vehicle agent. This can be subclassed with more complex models.
    This model assumes that the battery begins at 100% capacity and takes a linear amount of time to charge.
    Discharge happens when the robot moves, where the amount discharged is dependent on the
    speed at which the robot is moving.
    '''
    #these should be read from a config file
    speeds_ranges = [_ for _ in range(0, 41, 5)]
    flight_times = [6.3, 6.3, 6.15, 5.85, 5.3, 4.85, 4.6, 4.1, 3.6]
    #maximum flight range in metres (hover)
    maximum_range = 3000
    
    #recharge time from 0% - 100% in seconds (20 minutes)
    recharge_time = 60*20
    
    def __init__(self, intial_capacity_percentage):
        self.effective_capacities = [flight_time / max(RAVBattery.flight_times) for flight_time in RAVBattery.flight_times]
        #calculate effective flight ranges
        self.effective_flight_ranges = {speed: effective_capacity * RAVBattery.maximum_range for speed, effective_capacity in zip(RAVBattery.speeds_ranges, self.effective_capacities)}
        self.current_range = intial_capacity_percentage * RAVBattery.maximum_range 
        self.effective_capacity_given_speed_spline = splrep(RAVBattery.speeds_ranges, sorted(self.effective_flight_ranges.values(), reverse = True))
        self.predict_capacity_given_speed = lambda speed: splev(speed, self.effective_capacity_given_speed_spline)
        
    def plot_effective_range_vs_speed(self):
        plt.plot([speed for speed in range(40)], [self.predict_capacity_given_speed(speed) for speed in range(40)])
        plt.ylim(0)
            
    def get_remaining_range(self):
        return self.current_range
    
    def get_current_capacity(self):
        return self.current_range / RAVBattery.maximum_range
    
    def set_capacity(self, new_capacity):
        pass
    
    def reduce_capacity(self, amount):
        pass
    
    def increase_capacity(self, amount):
        pass
    
    def move_by_dist_at_speed(self, dist, speed) -> "True if the agent has enough battery capacity to move the required distance at the required speed, otherwise false":
        if self.current_range - (dist * (RAVBattery.maximum_range/self.predict_capacity_given_speed(speed)))> 0:
            #calculate the proportion of the battery would be used up relative to the maximum capacity travelling the given distance at the given speed
            #then scale up to the maximum range
            self.current_range = self.current_range - (dist * (RAVBattery.maximum_range/self.predict_capacity_given_speed(speed)))
            return True
        else:
            #no need to update
            return False
        
    def recharge_to_percentage(self, capacity_percentage) -> "The time taken to recharge the battery to the given percentage":
        self.current_range = capacity_percentage * RAVBattery.maximum_range 
        return RAVBattery.recharge_time * capacity_percentage

#%%

if __name__ == '__main__':
#%%    
    #set initial capacity as full
    test_battery = RAVBattery(1)
    test_battery.move_by_dist_at_speed(200, 5)
    print(test_battery.get_current_capacity())
    print(test_battery.get_remaining_range())
#%%
    print(test_battery.move_by_dist_at_speed(200, 12))
    print(test_battery.get_current_capacity())
    print(test_battery.get_remaining_range())
    test_battery.plot_effective_range_vs_speed()
#%%        
    test_battery.move_by_dist_at_speed(2000, 30)
    print(test_battery.get_current_capacity())
    print(test_battery.get_remaining_range())
    
    print(test_battery.move_by_dist_at_speed(280, 30))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    