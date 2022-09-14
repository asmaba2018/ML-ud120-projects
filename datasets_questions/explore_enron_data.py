#!/usr/bin/python3

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import joblib

enron_data = joblib.load(open("../final_project/final_project_dataset_unix.pkl", "rb"))

n_people = len(enron_data)
james_prentice = enron_data["PRENTICE JAMES"]["total_stock_value"]
wesley_colwell = enron_data["COLWELL WESLEY"]["from_this_person_to_poi"]
jeffrey_skilling = enron_data["SKILLING JEFFREY K"]["exercised_stock_options"]
poi_list = ['SKILLING JEFFREY K', 'LAY KENNETH L', 'FASTOW ANDREW S']
poi_counter = 0
salary_counter = 0
email_counter = 0
payment_counter = 0
poi_payment = 0

for people, features in enron_data.items():
    n_features = len(features.items())
    if features["total_payments"] == "NaN":
        payment_counter += 1
    if features["salary"] != "NaN":
        salary_counter += 1
    if features["email_address"] != "NaN":
        email_counter += 1
    if features["poi"] == 1:
        if features["total_payments"] == "NaN":
            poi_payment += 1
        poi_counter += 1

print("No. of people in enron data:", n_people)
print("No. of features in enron data:", n_features)
print("No. of persons of interest existing in enron data:", poi_counter)
print("Total stock value for James Prentice:", james_prentice)
print("No. of E-mails from Wesley Colwell to poi:", wesley_colwell)
print("Exercised stock options value for Jeffrey K Skilling:", jeffrey_skilling)

for person in poi_list:
    total_payment = enron_data[person]["total_payments"]
    print(f"{person} has total payments of: {total_payment}")

# print(enron_data["SKILLING JEFFREY K"]) #how to represent missing values 
print("No. of folks with identified salary:", salary_counter)
print("No. of folks with E-mail adresses:", email_counter)
print("No. of folks without a payment informations:", payment_counter, "with a percentage of:", payment_counter * 100/n_people)
print("No. persons of interest without a payment informations:", poi_payment, "with a percentage of:", poi_payment * 100/n_people)
