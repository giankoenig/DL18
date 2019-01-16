import pickle

def good_policies():
  good_policies = pickle.load(open('../optimal_policies_0.pol', "rb"))
  
  return good_policies
