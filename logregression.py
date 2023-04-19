from sklearn.linear_model import LogisticRegression

def main(data, labels):    
    model = LogisticRegression(max_iter=5000).fit(data, labels)
    return model