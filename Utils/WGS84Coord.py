# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 10:56:27 2018

@author: 13383861
"""
import math

#semi-major axis
WGSa = 6378137.0;
#semi-minor axis
WGSb = 6356752.314245;
#reciprocal of flattening
WGSf = 1 / 298.257223563; 
TwoPi = 2.0 * math.PI;

def vincenty

class WGS84Coordinate:
    
    def __init__(self, lat, long, alt):
        self.lat = lat
        self.long = long
        self.alt = alt
        
    def getDistanceToOther(other):
        if not isinstance(other, self):
            raise Exception("Can only get distance to another instance of WGS84Coordinate")
            
        else:
            
        