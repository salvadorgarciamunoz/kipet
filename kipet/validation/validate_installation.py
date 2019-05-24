#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import unittest
import sys
import os
import inspect
import subprocess

import imp

#try:
    #imp.find_module('casadi')
    #found_casadi = True
#except ImportError:
    #found_casadi = False

examplesMainDir = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(
        inspect.currentframe()))), '..','examples'))


class TestExamples(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        sys.path.append(examplesMainDir)
        self.std_out = open("test_tut_examples.log", "w")

    @classmethod
    def tearDownClass(self):
        sys.path.remove(examplesMainDir)
        self.std_out.close()

    def _schedule(self, examples_dir):
        examples_tutorial_dir = os.path.join(examples_dir)
        examples_tutorial=[f for f in os.listdir(examples_dir) if os.path.isfile(os.path.join(examples_tutorial_dir,f)) and (f=='Ex_2_with_SVD.py' or f=='Ex_2_estimation.py' or f=='Ex_2_estimation_conf.py' or f=='Ex_2_estimation_conf_k_aug.py' or f=='Ex_2_estimationfefactoryTempV.py' or f=='Ex_9_estimability_with_problem_gen.py' or f=='Ex_11_estimation_mult_exp.py')]
        
        #####################################
        ###Tutorial-Examples:##
        #####################################
        #else:
        flagpy = 0
        countpy = 0
        flagps = 0
        for f in examples_tutorial:
            flagpy = subprocess.call([sys.executable,os.path.join(examples_tutorial_dir,f),'1'],
                                stdout=self.std_out,
                                stderr=subprocess.STDOUT)
            if flagpy!=0 and f=='Ex_2_with_SVD.py': 
                print("running SVD:",f,"failed")
                countpy = countpy + 1
                flagpy=1
                flagps=1
                
            if flagpy!=0 and f=='Ex_2_estimation.py': 
                print("running ipopt:",f,"failed")
                countpy = countpy + 1
                flagpy=1
                flagps=1
                
            if flagpy!=0 and f=='Ex_2_estimation_conf.py': 
                print("running sipopt:",f,"failed")
                countpy = countpy + 1
                flagpy=1
                flagps=1
                
            if flagpy!=0 and f=='Ex_2_estimation_conf_k_aug.py': 
                print("running k_aug:",f,"failed")
                countpy = countpy + 1
                flagpy=1
                flagps=1
                
            if flagpy!=0 and f=='Ex_2_estimationfefactoryTempV.py': 
                print("running ipopt with fe_factory:",f,"failed")
                countpy = countpy + 1
                flagpy=1
                flagps=1
                
            if flagpy!=0 and f=='Ex_9_estimability_with_problem_gen.py': 
                print("running estimability:",f,"failed")
                countpy = countpy + 1
                flagpy=1
                flagps=1

            if flagpy!=0 and f=='Ex_11_estimation_mult_exp.py': 
                print("running estimability:",f,"failed")
                countpy = countpy + 1
                flagpy=1
                flagps=1
                
            if flagpy==0 and f=='Ex_2_with_SVD.py': 
                print("running SVD:",f,"passed")
                
            if flagpy==0 and f=='Ex_2_estimation.py': 
                print("running ipopt:",f,"passed")
                
            if flagpy==0 and f=='Ex_2_estimation_conf.py': 
                print("running sipopt:",f,"passed")
                
            if flagpy==0 and f=='Ex_2_estimation_conf_k_aug.py': 
                print("running k_aug:",f,"passed")
                
            if flagpy==0 and f=='Ex_2_estimationfefactoryTempV.py': 
                print("running ipopt with fe_factory:",f,"passed")
                
            if flagpy==0 and f=='Ex_9_estimability_with_problem_gen.py': 
                print("running estimability:",f,"passed")

            if flagpy==0 and f=='Ex_11_estimation_mult_exp.py': 
                print("running multiple experiments:",f,"passed")
            continue
        print(countpy,"of selected files in",examples_tutorial_dir,"failed")
        return {self.assertEqual(int(flagpy),0),self.assertEqual(int(flagps),0)}

    def test_tutorial_examples(self):
        print('##############Tutorial Problems to Test Installation#############')
        examples_dir = os.path.join(examplesMainDir)
        self._schedule(examples_dir)

    def run_all_examples(self):

        self.test_tutorial_examples()


    def runTest(self):
        print("hallo")  #: XD
        


if __name__ == '__main__':
    unittest.main()
