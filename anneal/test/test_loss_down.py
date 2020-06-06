import unittest
import anneal.utils
from anneal.models.HmmSimple import HmmSimple
from anneal.models.MixtureGaussian import MixtureGaussian
from anneal.models.MixtureGaussianDMP import MixtureGaussianDMP
from anneal.models.MixtureDirichlet import MixtureDirichlet
from anneal.models.HmmMIxtureRNA import HmmMixtureRNA
import os

LR = 0.05
ST = 2
ST2 = 2
data_dir = "anneal/data"

class LossTest(unittest.TestCase):
    def test_GaussianMixture(self):
        print(os.curdir)

        data_dict = anneal.utils.load_simulation_seg(data_dir, "example1")


        params, loss = anneal.utils.run_analysis(data_dict, MixtureGaussian, steps=ST, lr=LR)
        self.assertNotEqual(loss[1], loss[0])

        params, loss = anneal.utils.run_analysis(data_dict, MixtureGaussian, steps=ST, lr=LR, MAP=False)
        self.assertNotEqual(loss[1], loss[0])

        params, loss = anneal.utils.run_analysis(data_dict, MixtureGaussian, steps=ST, lr=LR, posteriors=True, step_post=ST2)
        self.assertNotEqual(loss[1], loss[0])

        params, loss = anneal.utils.run_analysis(data_dict, MixtureGaussian, steps=ST, lr=LR, MAP=False, posteriors=True, step_post=ST2)
        self.assertNotEqual(loss[1], loss[0])


    def test_GaussianMixtureDPM(self):
        data_dict = anneal.utils.load_simulation_seg(data_dir, "example1")

        params, loss = anneal.utils.run_analysis(data_dict, MixtureGaussianDMP, steps=ST, lr=LR)
        self.assertNotEqual(loss[1], loss[0])

        params, loss = anneal.utils.run_analysis(data_dict, MixtureGaussianDMP, steps=ST, lr=LR, MAP=False)
        self.assertNotEqual(loss[1], loss[0])

        params, loss = anneal.utils.run_analysis(data_dict, MixtureGaussianDMP, steps=ST, lr=LR,
                                                 posteriors=True, step_post=ST2)
        self.assertNotEqual(loss[1], loss[0])

        params, loss = anneal.utils.run_analysis(data_dict, MixtureGaussianDMP, steps=ST, lr=LR, MAP=False,
                                                 posteriors=True, step_post=ST2)
        self.assertNotEqual(loss[1], loss[0])

    def test_DirichletMixture(self):
        data_dict = anneal.utils.load_simulation_seg(data_dir, "example1")

        params, loss = anneal.utils.run_analysis(data_dict, MixtureDirichlet, steps=ST, lr=LR)
        self.assertNotEqual(loss[1], loss[0])

        params, loss = anneal.utils.run_analysis(data_dict, MixtureDirichlet, steps=ST, lr=LR,
                                                 MAP=False)
        self.assertNotEqual(loss[1], loss[0])

        params, loss = anneal.utils.run_analysis(data_dict, MixtureDirichlet, steps=ST, lr=LR,
                                                 posteriors=True, step_post=ST2)
        self.assertNotEqual(loss[1], loss[0])

        params, loss = anneal.utils.run_analysis(data_dict, MixtureDirichlet, steps=ST, lr=LR,
                                                 MAP=False,
                                                 posteriors=True, step_post=ST2)
        self.assertNotEqual(loss[1], loss[0])

    def test_HmmSimple(self):
        data_dict = anneal.utils.load_simulation_seg(data_dir, "example1")

        params, loss = anneal.utils.run_analysis(data_dict, HmmSimple, steps=ST, lr=LR)
        self.assertNotEqual(loss[1], loss[0])



    def test_HmmMixtureRNA(self):
        data_dict = anneal.utils.load_simulation_seg(data_dir, "example1")

        params, loss = anneal.utils.run_analysis(data_dict, HmmMixtureRNA, steps=ST, lr=LR)
        self.assertNotEqual(loss[1], loss[0])

        params, loss = anneal.utils.run_analysis(data_dict, HmmMixtureRNA, steps=ST, lr=LR,
                                                 MAP=False)
        self.assertNotEqual(loss[1], loss[0])

        params, loss = anneal.utils.run_analysis(data_dict, HmmMixtureRNA, steps=ST, lr=LR,
                                                 posteriors=True, step_post=ST2)
        self.assertNotEqual(loss[1], loss[0])

        params, loss = anneal.utils.run_analysis(data_dict, HmmMixtureRNA, steps=ST, lr=LR,
                                                 MAP=False,
                                                 posteriors=True, step_post=ST2)
        self.assertNotEqual(loss[1], loss[0])



if __name__ == '__main__':
    unittest.main()
