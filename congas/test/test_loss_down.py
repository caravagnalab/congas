import unittest
import congas.utils
from congas.models import MixtureGaussian
import os

LR = 0.05
ST = 2
ST2 = 2
data_dir = "congas" + os.sep + "data"

class LossTest(unittest.TestCase):
    def test_GaussianMixture(self):
        print(os.curdir)

        data_dict = congas.utils.load_simulation_seg(data_dir, "example1")


        params, loss = congas.utils.run_analysis(data_dict, MixtureGaussian, steps=ST, lr=LR)
        self.assertNotEqual(loss[1], loss[0])



    #def test_GaussianMixtureDPM(self):
     #   data_dict = congas.utils.load_simulation_seg(data_dir, "example1")

    #    params, loss = congas.utils.run_analysis(data_dict, MixtureGaussianDMP, steps=ST, lr=LR)
     #   self.assertNotEqual(loss[1], loss[0])

     #   params, loss = congas.utils.run_analysis(data_dict, MixtureGaussianDMP, steps=ST, lr=LR, MAP=False)
    #    self.assertNotEqual(loss[1], loss[0])

     #   params, loss = congas.utils.run_analysis(data_dict, MixtureGaussianDMP, steps=ST, lr=LR,
      #                                            step_post=ST2)
    #    self.assertNotEqual(loss[1], loss[0])

    #    params, loss = congas.utils.run_analysis(data_dict, MixtureGaussianDMP, steps=ST, lr=LR, MAP=False,
     #                                            step_post=ST2)
      #  self.assertNotEqual(loss[1], loss[0])





if __name__ == '__main__':
    unittest.main()
