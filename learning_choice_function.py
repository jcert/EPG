import numpy as np

from scipy.special import softmax
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV


if (__name__ == '__main__'):
    eta = 1
    number_of_strategy = 15
    strategies_available = np.arange(0, number_of_strategy)

    ## Generate dataset
    N_data_samples = 100000
    rewards = np.random.rand(N_data_samples, number_of_strategy)
    strategy_selections = [np.random.choice(a=strategies_available,
                                            size=1,
                                            p=softmax(reward/eta))[0] for reward in rewards]

    ## Data fitting
    base_clf = GaussianNB()
    calibrated_clf = CalibratedClassifierCV(base_clf, cv=3)
    calibrated_clf.fit(rewards, strategy_selections)

    ## Validation
    for _ in range(5):
        reward = np.random.random(number_of_strategy)
        estimated_choice_probability = calibrated_clf.predict_proba([reward])[0]
        logit_choice_probability = softmax(reward/eta)

        print(reward)
        fig = plt.figure(figsize=(6,4))
        ax = fig.add_subplot(111)
        ax.plot(logit_choice_probability, 'r', linewidth=5, label='estimated probaility')
        ax.plot(estimated_choice_probability, 'b--', label='logit')
        ax.set_xticks(range(number_of_strategy),
                      ('s{}'.format(s+1) for s in range(number_of_strategy)))

        plt.legend()
        plt.show()

