#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = [(age.item(), worth.item(), abs((pre - worth).item()))
            for age, worth, pre in zip(ages, net_worths, predictions)]
    
    ### your code goes here
    

    
    return sorted(cleaned_data, key = lambda x : x[-1], reverse = True)[9:]

