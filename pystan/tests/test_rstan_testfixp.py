import unittest

import numpy as np

import pystan


class TestFixedParam(unittest.TestCase):

    def test_fixedp(self):
        code = """
            model {
            }

            generated quantities {
            real y;
            y <- normal_rng(0, 1);
            }
            """
        self.assertRaises(RuntimeError, pystan.stan, model_code=code)

        fit = pystan.stan(model_code=code, iter=1000, chains=1, algorithm='Fixed_param')
        extr = fit.extract()
        self.assertTrue(-10 < np.mean(extr['y']) < 10)
        self.assertTrue(0 < np.std(extr['y']) < 10)

        fit2 = pystan.stan(fit=fit, iter=1000, chains=1, algorithm='Fixed_param')
        extr = fit2.extract()
        self.assertTrue(-10 < np.mean(extr['y']) < 10)
        self.assertTrue(0 < np.std(extr['y']) < 10)
