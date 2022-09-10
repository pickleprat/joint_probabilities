
import pandas as pd 
import numpy as np 
from scipy.integrate import quad
import sympy as smp 
 


class ContinousProbability:
    X_range: list 
    Y_range: list
    x: smp.symbols
    y: smp.symbols
    phi: smp.functions
    

    def __init__(self, phi, X_range, Y_range):
        self.phi = phi
        self.X_range = X_range
        self.Y_range = Y_range
        self.x, self.y = smp.symbols("x y", real=True)
        

        if not self.validity():
            raise ValueError("The function is not valid.")
    

    def integrate(self, X:list, Y:list):
        return smp.integrate(self.phi, (self.x, X[0], X[1]), (self.y, Y[0], Y[1]))

    def validity(self):
        if self.integrate(self.X_range, self.Y_range) == 1:
            return True 
        return False

    def margX(self, X: list):

        if X == None:
            return smp.integrate(self.phi, (self.y, self.Y_range[0], self.Y_range[1]))

        elif ((X[0] >= self.X_range[0]) and (X[1] <= self.X_range[1])):
            return smp.integrate(self.phi, (self.x, X[0], X[1]), (self.y, self.Y_range[0], self.Y_range[1]))

    def margY(self, Y: list):
    
        if Y == None:
            return smp.integrate(self.phi, (self.x, self.X_range[0], self.X_range[1]))

        elif ((Y[0] >= self.Y_range[0]) and (Y[1] <= self.Y_range[1])):
            return smp.integrate(self.phi, (self.y, Y[0], Y[1]), (self.x, self.X_range[0], self.X_range[1]))

    def Ex(self):
        return smp.integrate(self.x*self.margX(None), (self.x, self.X_range[0], self.X_range[1]))

    def Ey(self):
        return smp.integrate(self.y*self.margY(None), (self.y, self.Y_range[0], self.Y_range[1]))

    def VarX(self):
        return smp.integrate( (self.x**2 )*self.margX(None), (self.x, self.X_range[0], self.X_range[1])) - self.Ex()**2

    def VarY(self):
        return smp.integrate( (self.y**2 )*self.margY(None), (self.y, self.Y_range[0], self.Y_range[1])) - self.Ey()**2

    def condX(self, Y: list, givenX: list):
        return self.integrate(givenX, Y)/self.margX(givenX)

    def condY(self, X: list, givenY: list):
        return self.integrate(X, givenY)/self.margY(givenY)

    def Cov(self):
        return smp.integrate(self.x*self.y*self.phi, (self.x, self.X_range[0], self.X_range[1]), (self.y, self.Y_range[0], self.Y_range[1])) - self.Ex()*self.Ey()

    def stdX(self):
        return (self.VarX())**0.5

    def stdY(self):
        return (self.VarY())**0.5

    def __repr__(self):
        details = [
            f"The expected Value of X : {self.Ex().evalf()}", 
            f"The expected Value of Y : {self.Ey().evalf()}", 
            f"The Variance Value of X : {self.VarX().evalf()}", 
            f"The Variance Value of Y : {self.VarY().evalf()}", 
            f"The standard Devit of X : {self.stdX().evalf()}", 
            f"The standard Devit of Y : {self.stdY().evalf()}",
            f"The Covariance of X Y   : {self.Cov().evalf()}"
        ]
        return "\n".join(details)



    

    
        
