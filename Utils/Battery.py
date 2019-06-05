# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 16:08:00 2019

@author: 13383861

Simulates a battery
"""

import time
import typing
import asyncio
import threading
from enum import Enum

from scipy.interpolate import splrep, splev
import matplotlib.pyplot as plt

from Utils import Vector3r

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
class RAVBatterySimulator:
    
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
    #recharge_time = 60*20
    recharge_time = 10
    
    def __init__(self, intial_capacity_percentage):
        self.effective_capacities = [flight_time / max(RAVBatterySimulator.flight_times) for flight_time in RAVBatterySimulator.flight_times]
        #calculate effective flight ranges
        self.effective_flight_ranges = {speed: effective_capacity * RAVBatterySimulator.maximum_range for speed, effective_capacity in zip(RAVBatterySimulator.speeds_ranges, self.effective_capacities)}
        self.current_range = intial_capacity_percentage * RAVBatterySimulator.maximum_range 
        self.effective_capacity_given_speed_spline = splrep(RAVBatterySimulator.speeds_ranges, sorted(self.effective_flight_ranges.values(), reverse = True))
        self.predict_capacity_given_speed = lambda speed: splev(speed, self.effective_capacity_given_speed_spline)
        self.initial_capacity = intial_capacity_percentage
        self.current_capacity = intial_capacity_percentage
        self._is_recharging = False
        self.__charging_start_time = None
        self.__current_recharge_time = None
        
    def poll(self):
        '''A poll method updates the current state of the battery - whether it is recharging or not. This should ideally be called at each timestep 
        while the agent is in operation'''
        self.is_recharging()
        
    def plot_effective_range_vs_speed(self):
        plt.plot([speed for speed in range(40)], [self.predict_capacity_given_speed(speed) for speed in range(40)])
        plt.ylim(0)
            
    def get_remaining_range_at_speed(self, speed):
        '''Returns how far the RAV is predicted to fly at the given speed'''
        return self.predict_capacity_given_speed(speed) #* self.current_range
    
    def get_current_capacity(self):
        return self.current_capacity
    
    def set_capacity(self, new_capacity):
        self.current_capacity = new_capacity
    
    def reduce_capacity(self, amount):
        self.current_capacity -= amount
    
    def increase_capacity(self, amount):
        self.current_capacity += amount
    
    def move_by_dist_at_speed(self, dist, speed) -> "True if the agent has enough battery capacity to move the required distance at the required speed, otherwise false":
        if self.current_range - (dist * (RAVBatterySimulator.maximum_range/self.predict_capacity_given_speed(speed)))> 0:
            #calculate the proportion of the battery would be used up relative to the maximum capacity travelling the given distance at the given speed
            #then scale up to the maximum range
            self.current_range = self.current_range - (dist * (RAVBatterySimulator.maximum_range/self.predict_capacity_given_speed(speed)))
            self.current_capacity = self.current_range / RAVBatterySimulator.maximum_range
            return True
        else:
            #no need to update
            return False
        
    def _recharge_for_n_seconds(self, number_of_seconds):
        self._is_recharging = True
        #sets the amount of time the battery needs to recharge
        self.__current_recharge_time = number_of_seconds
        #records when the battery has begun to start recharging
        self.__charging_start_time = time.time()
        
    def _stop_recharging(self):
        self._is_recharging = False
        
    def is_recharging(self):
        is_recharging = (time.time() - self.__charging_start_time) < self.__current_recharge_time
        
        if self._is_recharging and not is_recharging:
            #this means that the battery is meant to be recharging but time has elapsed since it started recharging
            self._stop_recharging()
            self.current_capacity = self._desired_capacity
            self._desired_capacity = None
            
        if not is_recharging:
            self._stop_recharging()
        else:
            #update the battery's current capacity 
            #the proportion of time the battery has spent charging until it reaches desired capacity is:
            (time.time() - self.__charging_start_time)/self.__current_recharge_time
            self.current_capacity = self._initial_recharge_capacity + ((self._desired_capacity - self._initial_recharge_capacity) *  ((time.time() - self.__charging_start_time)/self.__current_recharge_time))
            self.current_range = self.current_capacity * RAVBatterySimulator.maximum_range
        return is_recharging
        
    def recharge_to_percentage(self, capacity_percentage) -> "The time taken to recharge the battery to the given percentage":
        '''Assumes recharge time is a linear with respect to battery capacity. If the battery is already recharging, return
        None and let it continue recharging. It is possible to cancel the recharging and then start it again.'''
        if self._is_recharging:
            return None
        #if the battery is already above this percentage, can't recharge to it
        if self.current_capacity > capacity_percentage:
            #0 seconds to recharge to a capacity less than the current capacity
            return 0
        self._is_recharging = True
        self._initial_recharge_capacity = self.current_capacity
        self._desired_capacity = capacity_percentage
        #self.current_capacity = capacity_percentage
        recharge_time = RAVBatterySimulator.recharge_time * capacity_percentage
        #don't return until the battery has recharged
        self._recharge_for_n_seconds(recharge_time)
        return RAVBatterySimulator.recharge_time * capacity_percentage
    
    def cancel_recharging(self):
        self.poll()
        self._is_recharging = False
        self.__current_recharge_time = 0
        self.poll()
        
    
    def reset(self):
        '''Resets the RAV battery to its initial state'''
        self.__init__(self.initial_capacity)
        
    def generate_capacity_readings_every_second(self, operational_speed, out_file_location = None):
        '''
        Generates an array of readings taken at 1s time intervals to be used to train battery hmm
        '''
        self.recharge_to_percentage(100)
        capacities = []
        #move for 1 second as often as possible
        while test_battery.get_remaining_range_at_speed(operational_speed) > 0:
            if not test_battery.move_by_dist_at_speed(operational_speed, operational_speed):
                break
            capacities.append(self.get_current_capacity())
        if out_file_location:
            with open(out_file_location,'w') as f_object:
                #write the csv header
                f_object.write('time_index' + ',' + 'battery_capacity' + '\n')
                for time_stamp_index, capacity in enumerate(capacities):
                    #write each line of csv
                    f_object.write(str(time_stamp_index) + ',' + str(capacity) + '\n')
        return capacities

class RAVBatterySimulatorAgent:
    '''
    A class that manages a physical RAV's battery model.
    '''
    
    def __init__(self, initial_battery_percentage, initial_location: Vector3r):
        self.battery = RAVBatterySimulator(initial_battery_percentage)
        self.current_location = initial_location
        
    def can_move_to_location(self, target_location: Vector3r, current_location: Vector3r, speed: float):
        '''Returns boolean whether or not agent can move to the desired location at the desired speed with it's remaining battery capacity'''
        return self.battery.get_remaining_range_at_speed() > target_location.distance_to(current_location)
        
    def get_nearest_feasible_location(self, target_location: Vector3r, current_location: Vector3r, speed: float):
        '''Returns the nearest feasible location to move to given a desired target location'''
        if self.battery.current_capacity == 0:
            return self.current_location
        if not self.can_move_to_location(target_location, current_location, speed):
            #send the agent to the point nearest the target location and let it die!
            slope = (target_location.y_val - current_location.y_val) / (target_location.x_val - current_location.x_val) 
            slope_squared = slope **2
            nearest_point_y = (((self.battery.get_remaining_range_at_speed(speed)**2) * (slope_squared)) / (slope_squared + 1))**0.5
            nearest_point_x = nearest_point_y/slope
            target_location = Vector3r(current_location.x_val + nearest_point_x, current_location.y_val + nearest_point_y, current_location.z_val)
        #self.battery.move_by_dist_at_speed(target_location.distance_to(current_location), speed)
        return target_location
    
    def move_to_location(self, target_location: Vector3r, current_location: Vector3r, speed: float):
        '''
        Given the order to move to a location, returns the closest location to the target which the agent can reach
        from it's current location and updates the battery accordingly. If the battery is busy recharging, returns
        None.
        '''
        if self.battery.is_recharging():
            return None
        nearest_feasible_location = self.get_nearest_feasible_location(target_location, current_location, speed)
        self.battery.poll()
        self.battery.move_by_dist_at_speed(nearest_feasible_location, speed)
        self.current_location = nearest_feasible_location
    
    def recharge_to_percentage(self, capacity_percentage):
        self.battery.poll()
        self.battery.recharge_to_percentage(capacity_percentage)
        
    def battery_is_recharging(self):
        self.battery.poll()
        return self.battery.is_recharging()
    
    def cancel_battery_recharging(self):
        self.battery.cancel_recharging()
        
    
            
#%%

if __name__ == '__main__':
    
#%% 
    import math
    #set initial capacity as full
    test_battery = RAVBatterySimulator(1)
#    test_battery.move_by_dist_at_speed(200, 5)
#    print(test_battery.get_current_capacity())
#    print(test_battery.get_remaining_range_at_speed(5))
#    print(test_battery.get_remaining_range_at_speed(8))
    data = test_battery.generate_capacity_readings_every_second(4, "D:\\OccupancyGrid\\Data\\BatteryData\\SimulatedBatteryData.csv")
    binned_values = [round(_ * 10 ) for _ in data]
    print(binned_values)
#%%
    print(test_battery.move_by_dist_at_speed(200, 12))
    print(test_battery.get_current_capacity())
    print(test_battery.get_remaining_range_at_speed(6))
    test_battery.plot_effective_range_vs_speed()
#%%        
    test_battery.move_by_dist_at_speed(2000, 30)
    print(test_battery.get_current_capacity())
    print(test_battery.get_remaining_range_at_speed(2))
    
    print(test_battery.move_by_dist_at_speed(280, 30))
#%%    
    print(test_battery.get_current_capacity())
    seconds_to_recharged = test_battery.recharge_to_percentage(0.9)
    print("number of seconds until battery is recharged: ", seconds_to_recharged)
    assert test_battery.is_recharging()
    time.sleep(seconds_to_recharged/2)
    assert test_battery.is_recharging()
    time.sleep(seconds_to_recharged/2)
    test_battery.poll()
    assert not test_battery.is_recharging()
    assert test_battery.get_current_capacity() == 0.9
    
#%%
    print(test_battery.get_current_capacity())
    test_battery.move_by_dist_at_speed(500, 35)
    current_cap = test_battery.get_current_capacity()
    print(current_cap)
    seconds_to_recharged = test_battery.recharge_to_percentage(0.95)
    print("number of seconds until battery is recharged: ", seconds_to_recharged)
    assert test_battery.is_recharging()
    time.sleep(seconds_to_recharged)

    test_battery.cancel_recharging()
    assert not test_battery.is_recharging()
    assert math.isclose(test_battery.get_current_capacity(), 0.95, rel_tol = 0.01)

#%%
    #test that charging for half time it would take to charge to desired capacity results in adding half capacity that
    #would have needed to be added to charge to desired capacity
    print(test_battery.get_current_capacity())
    test_battery.move_by_dist_at_speed(500, 35)
    current_cap = test_battery.get_current_capacity()
    print(current_cap)
    seconds_to_recharged = test_battery.recharge_to_percentage(0.95)
    print("number of seconds until battery is recharged: ", seconds_to_recharged)
    assert test_battery.is_recharging()
    time.sleep(seconds_to_recharged/2)
    test_battery.cancel_recharging()
    assert not test_battery.is_recharging()
    assert math.isclose(test_battery.get_current_capacity(), current_cap + (0.95 - current_cap)/2, rel_tol = 0.01)

#%%    
   
    import math
    assert math.isclose(test_battery.get_current_capacity(), current_cap + (0.95 - current_cap)/2, rel_tol = 0.01)
    test_battery.recharge_to_percentage(0.95)
    test_battery.poll()
    assert test_battery.get_current_capacity() == 0/95
    
    #%%
    assert test_battery.get_current_capacity() == 0.9  
    assert test_battery.get_remaining_range_at_speed(6) > test_battery.get_remaining_range_at_speed(9) > test_battery.get_remaining_range_at_speed(12)
    
    
    #%%
    bat_agent = RAVBatterySimulatorAgent(1, Vector3r(0,0))
    bat_agent.can_move_to_location()
    
    #%%
    
    
    