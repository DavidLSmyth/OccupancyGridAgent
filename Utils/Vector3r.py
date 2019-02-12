# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 11:02:10 2018

@author: 13383861
"""

# =============================================================================
# class Vector3r:
#     '''A coordinate which represents an objects location in an unreal engine environment'''
#     def __init__(self, x_val, y_val, z_val = 0):
#         self.x_val, self.y_val, self.z_val = x_val, y_val, z_val
#         if not isinstance(self.x_val, float):
#             try:
#                 self.x_val = float(self.x_val)
#             except Exception as e:
#                 raise(e)
#         
#         if not isinstance(self.y_val, float):
#             try:
#                 self.y_val = float(self.y_val)
#             except Exception as e:
#                 raise(e)
#                 
#         if not isinstance(self.z_val, float):
#             try:
#                 self.z_val = float(self.z_val)
#             except Exception as e:
#                 raise(e)
#                 
#     def to_vector3r(self):
#         return Vector3r(self.x_val, self.y_val, self.z_val)
#         
#     def __add__(self, other):
#         return Vector3r(self.x_val + other.x_val, self.y_val + other.y_val, self.z_val + other.z_val)
#         
#     def __sub__(self, other):
#         return Vector3r(self.x_val - other.x_val, self.y_val - other.y_val, self.z_val - other.z_val)
#         
#     def mul(self, int):
#         pass
#     
#     def get_dist_to_other(self, other):
#         return ((self.x_val - other.x_val)**2 + (self.y_val - other.y_val)**2 + (self.z_val - other.z_val)**2)**0.5
#     
#     def __eq__(self, other):
#         if self.x_val == other.x_val and self.y_val == other.y_val and self.z_val == other.z_val:
#             return True
#         else:
#             return False
#     
#     def __str__(self):
#         return 'Vector3r({x_val},{y_val},{z_val})'.format(x_val = self.x_val, y_val = self.y_val, z_val = self.z_val)
#     
#     def __repr__(self):
#         return 'Vector3r({x_val},{y_val},{z_val})'.format(x_val = self.x_val, y_val = self.y_val, z_val = self.z_val)
#     
#     def __hash__(self):
#          return hash(repr(self))
# =============================================================================
     
import numpy as np
class Vector3r():
#    x_val = 0.0
#    y_val = 0.0
#    z_val = 0.0

    def __init__(self, x_val = 0.0, y_val = 0.0, z_val = 0.0):
        #not sure whether converting to float is a good idea here or not
        #needed to modify this because although Vector3r(10,0,0) == Vector3r(10.0, 0, 0) returns True
        #looking up a dictionary containing Vector3r(10,0,0) for the value Vector3r(10.0,0.0,0.0) returns a KeyError
        try:
            self.x_val = float(x_val)
            self.y_val = float(y_val)
            self.z_val = float(z_val)
            
        except Exception as e:
            print("Error in converting x, y, z components to float")
            raise e
        
    def _verify_floats(self):
        for dimension in [self.x_val, self.y_val, self.z_val]:
            if not isinstance(dimension, float):
                return False
        return True

    @staticmethod
    def nanVector3r():
        return Vector3r(np.nan, np.nan, np.nan)
    
    def __str__(self):
        return f"Vector3r({self.x_val}, {self.y_val}, {self.z_val})"

    def __add__(self, other):
        return Vector3r(self.x_val + other.x_val, self.y_val + other.y_val, self.z_val + other.z_val)

    def __sub__(self, other):
        return Vector3r(self.x_val - other.x_val, self.y_val - other.y_val, self.z_val - other.z_val)

    def __truediv__(self, other):
        if type(other) in [int, float] + np.sctypes['int'] + np.sctypes['uint'] + np.sctypes['float']:
            return Vector3r( self.x_val / other, self.y_val / other, self.z_val / other)
        else: 
            raise TypeError('unsupported operand type(s) for /: %s and %s' % ( str(type(self)), str(type(other))) )

    def __mul__(self, other):
        if type(other) in [int, float] + np.sctypes['int'] + np.sctypes['uint'] + np.sctypes['float']:
            return Vector3r(self.x_val*other, self.y_val*other, self.z_val)
        else: 
            raise TypeError('unsupported operand type(s) for *: %s and %s' % ( str(type(self)), str(type(other))))
            
    def __eq__(self, other):
        '''3d vectors are equal if x,y,z components are all equal'''
        return all([self.x_val == other.x_val, self.y_val == other.y_val, self.z_val == other.z_val])

    def dot(self, other):
        if type(self) == type(other):
            return self.x_val*other.x_val + self.y_val*other.y_val + self.z_val*other.z_val
        else:
            raise TypeError('unsupported operand type(s) for \'dot\': %s and %s' % ( str(type(self)), str(type(other))) )

    def cross(self, other):
        if type(self) == type(other):
            cross_product = np.cross(self.to_numpy_array(), other.to_numpy_array)
            return Vector3r(cross_product[0], cross_product[1], cross_product[2])
        else:
            raise TypeError('unsupported operand type(s) for \'cross\': %s and %s' % ( str(type(self)), str(type(other))) )

    def get_length(self):
        return ( self.x_val**2 + self.y_val**2 + self.z_val**2 )**0.5

    def distance_to(self, other):
        return ( (self.x_val-other.x_val)**2 + (self.y_val-other.y_val)**2 + (self.z_val-other.z_val)**2 )**0.5

    def to_Quaternionr(self):
        return Quaternionr(self.x_val, self.y_val, self.z_val, 0)

    def to_numpy_array(self):
        return np.array([self.x_val, self.y_val, self.z_val], dtype=np.float32)
    
   
    def __repr__(self):
        return 'Vector3r({x_val},{y_val},{z_val})'.format(x_val = self.x_val, y_val = self.y_val, z_val = self.z_val)
    
    def __hash__(self):
         return hash(repr(self))

#%%     
if __name__ == '__main__':
    v1 = Vector3r(2, 3, 4)
    v2 = Vector3r(float(2), float(3), float(4))
    v3 = Vector3r(int(2), int(3))
    v4 = Vector3r(float(2), float(3))
    grid = {v1: 2, v3: 3}
    assert v2 in grid    
    assert v4 in grid
    grid[v2]
    print(v1.__repr__())
    print(v2.__repr__())


