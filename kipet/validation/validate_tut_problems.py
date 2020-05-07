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
        examples_tutorial=[f for f in os.listdir(examples_dir) if os.path.isfile(os.path.join(examples_tutorial_dir,f)) and f.endswith('.py')]
        number_of_examples = len(examples_tutorial)
        #####################################
        ###Tutorial-Examples:##
        #####################################
        #else:
        flagpy = 0
        countpy = 0
        flagps = 0
        for n, f in enumerate(examples_tutorial):
            print(f'Running tutorial example ({n + 1}/{number_of_examples}): {f}')
            
            flagpy = subprocess.call([sys.executable,os.path.join(examples_tutorial_dir,f),'1'],
                                stdout=self.std_out,
                                stderr=subprocess.STDOUT)
            if flagpy!=0: 
                print(f"\n\t #### {f} FAILED ####\n")
                countpy = countpy + 1
                flagpy=1
                flagps=1

            else:
                print(f"\n\t #### {f} PASSED ####\n")
            continue
        print(countpy,"files in",examples_tutorial_dir,"failed")
        return {self.assertEqual(int(flagpy),0),self.assertEqual(int(flagps),0)}

    def test_tutorial_examples(self):
        print('##############Tutorial Problems#############')
        examples_dir = os.path.join(examplesMainDir)
        self._schedule(examples_dir)

    def run_all_examples(self):

        self.test_tutorial_examples()


    def runTest(self):
        print("hallo")  #: XD
        


if __name__ == '__main__':
    unittest.main()
