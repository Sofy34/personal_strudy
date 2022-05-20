from mlxtend.classifier import StackingCVClassifier

def ensemble(classifiers,meta_classifier):
    return StackingCVClassifier(classifiers=classifiers,meta_classifier= meta_classifier,random_state=50)
    