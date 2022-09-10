#Author: Pratyush Pradeep Rao 
#Date: 07/09/2022

import numpy as np 
import pandas as pd 

class DiscreteProbability:

    X_vals: list
    Y_vals: list 

    def __init__(self, phi, X_vals:list, Y_vals:list) :
        self.phi = phi
        self.X_vals = X_vals
        self.Y_vals = Y_vals
    
        if  not self.validity():
            raise ValueError("Invalid probability function or range of random variables.")
   
    def validity(self):
        area = 0 
        for X in self.X_vals:
            for Y in self.Y_vals:
                area = area + self.phi(X, Y)
        if area == 1: return True 
        else: return False 
    

    def margX(self, X):
        if X not in self.X_vals:
            raise ValueError("The X value is invalid.")
        marg = 0 
        for Y in self.Y_vals:
            marg = marg + self.phi(X, Y)
        return marg

    def margY(self, Y):
        if Y not in self.Y_vals:
            raise ValueError("The Y value is invalid.")
        marg = 0 
        for X in self.X_vals:
            marg = marg + self.phi(X, Y)
        return marg

    def Ex(self):
        mean = 0 
        for X in self.X_vals:
            mean = mean + X*self.margX(X)
        return mean 

    def Ey(self):
        mean = 0 
        for Y in self.Y_vals:
            mean = mean + Y*self.margY(Y)
        return mean

    def VarX(self):
        var = 0
        for X in self.X_vals:
            var = (X**2)*self.margX(X) + var
            
        return var  -  self.Ex()**2

    def VarY(self):
        var = 0
        for Y in self.Y_vals:
            var = (Y**2)*self.margY(Y) + var
            
        return var  -  self.Ey()**2

    def condX(self, Y, givenX):
        if Y not in self.Y_vals and givenX not in self.X_vals:
            raise ValueError("Values are invalid.")

        return self.phi(givenX, Y)/self.margX(givenX)

    def condY(self, X, givenY):
        if X not in self.X_vals and givenY not in self.Y_vals:
            raise ValueError("Values are invalid.")

        return self.phi(X, givenY)/self.margY(givenY)

    def stdX(self):
        return np.sqrt(self.VarX())

    def stdY(self):
        return np.sqrt(self.VarY())


    def condDistX(self):
        condX = {f"Y = {y}": [self.condX(y, givenX) for givenX in self.X_vals] for y in self.Y_vals}
        return pd.DataFrame(condX, index=[f"given X = {x}" for x in self.X_vals])

    def condDistY(self):
        condY = {f"X = {x}":[self.condY(x, givenY) for givenY in self.Y_vals] for x in self.X_vals}
        return pd.DataFrame(condY, index=[f"given Y = {y}" for y in self.Y_vals])

    def margDistX(self):
        margX = {f"X = {X}":self.margX(X) for X in self.X_vals }
        return pd.DataFrame(margX, index=["f x (X = x) = "])

    def margDistY(self):
        margY = {f"Y = {Y}": self.margY(Y) for Y in self.Y_vals}
        return pd.DataFrame(margY, index=["f y (Y = y) = "])

    def Cov(self):
        covar = 0 
        for X in self.X_vals:
            for Y in self.Y_vals:
                covar = covar + self.phi(X, Y)*X*Y
        return covar - self.Ex()*self.Ey()


    def __repr__(self):
        details = [
            f"The expected Value of X : {self.Ex()}", 
            f"The expected Value of Y : {self.Ey()}", 
            f"The Variance Value of X : {self.VarX()}", 
            f"The Variance Value of Y : {self.VarY()}", 
            f"The standard Devit of X : {self.stdX()}", 
            f"The standard Devit of Y : {self.stdY()}",
            f"The Covariance of X Y   : {self.Cov()}"
        ]

        return "\n".join(details)



    


    


    

    