#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import unittest
import sys
import os
import inspect
import subprocess

import imp

try:
    imp.find_module('casadi')
    found_casadi = True
except ImportError:
    found_casadi = False

examplesMainDir = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(
        inspect.currentframe()))), '..', 'examples'))


class TestExamples(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        sys.path.append(examplesMainDir)
        self.std_out = open("test_examples.log", "w")

    @classmethod
    def tearDownClass(self):
        sys.path.remove(examplesMainDir)
        self.std_out.close()

    def _schedule(self, examples_dir):
        examples_plainpyomo_dir = os.path.join(examples_dir)
        examples_plainpyomo=[f for f in os.listdir(examples_dir) if os.path.isfile(os.path.join(examples_plainpyomo_dir,f)) and f.endswith('.py')]

        if os.path.isdir(os.path.join(examples_dir,'pyomo'))==True:
            
            examples_pyomo_dir = os.path.join(examples_dir,'pyomo')
            examples_pyomo = [f for f in os.listdir(examples_pyomo_dir) if os.path.isfile(os.path.join(examples_pyomo_dir,f)) and f.endswith('.py')]
        
            #########################
            ##Pyomo-Examples:########
            #########################
            flag = 0
            count = 0
            for f in examples_pyomo:
                # print "running pyomo:",f
                flag = subprocess.call([sys.executable, os.path.join(examples_pyomo_dir, f), '1'],
                                       stdout=self.std_out,
                                       stderr=subprocess.STDOUT)
                if flag != 0:
                    print("running pyomo:", f, "failed")
                    count = count + 1
                    flag=1
                else:
                    print("running pyomo:", f, "passed")
                continue
            #####################################
            #####Casadi- and sipopt-Examples:####
            #####################################
            if found_casadi or os.path.isdir(os.path.join(examples_pyomo_dir,'sipopt')):
            ######################################
            #########sipopt-Files#################
            ######################################
                flags=0
                counts=0
                if os.path.isdir(os.path.join(examples_pyomo_dir,'sipopt'))==True:
                    examples_sipopt_dir = os.path.join(examples_pyomo_dir,'sipopt')
                    examples_sipopt = [g for g in os.listdir(examples_sipopt_dir) if os.path.isfile(os.path.join(examples_sipopt_dir,g)) and f.endswith('.py')]
                    for g in examples_sipopt:
                        # print "running casadi:",f
                        flags = subprocess.call([sys.executable,os.path.join(examples_sipopt_dir,f),'1'],
                                            stdout=self.std_out,
                                            stderr=subprocess.STDOUT)
                        if flags!=0:
                            print("running sipopt:",g,"failed")
                            counts = counts + 1
                            flags=1
                        else:
                            print("running sipopt:",g,"passed")
                        continue
                    print(counts," files in ",examples_sipopt_dir," failed")
                #########################
                #####Casadi-Examples:####
                #########################
                if os.path.isdir(os.path.join(examples_dir,'casadi'))==True:
                    examples_casadi_dir = os.path.join(examples_dir,'casadi')
                    examples_casadi = [f for f in os.listdir(examples_casadi_dir) if os.path.isfile(os.path.join(examples_casadi_dir,f)) and f.endswith('.py')]
                    countc = 0
                    flagc = 0
                    for f in examples_casadi:
                        # print "running casadi:",f
                        flagc = subprocess.call([sys.executable,os.path.join(examples_casadi_dir,f),'1'],
                                            stdout=self.std_out,
                                            stderr=subprocess.STDOUT)
                        if flagc!=0:
                            print("running casadi:",f,"failed")
                            countc = countc + 1
                            flagc=1
                        else:
                            print("running casadi:",f,"passed")
                        continue
                    print(countc,"files in",examples_casadi_dir,"failed")
            print(count," files in ",examples_pyomo_dir," failed")
            return {self.assertEqual(int(flagc),0), self.assertEqual(int(flags),0),self.assertEqual(int(flag),0)}
        
        ####################################
        ##Plain_Pyomo- and Paper-Examples:##
        ####################################
        else:
            flagpy = 0
            countpy = 0
            flagps = 0
            for f in examples_plainpyomo:
                flagpy = subprocess.call([sys.executable,os.path.join(examples_plainpyomo_dir,f),'1'],
                                    stdout=self.std_out,
                                    stderr=subprocess.STDOUT)
                if flagpy!=0: 
                    print("running plainpyomo or paper:",f,"failed")
                    countpy = countpy + 1
                    flagpy=1
                    flagps=1

                else:
                    print("running plainpyomo or paper:",f,"passed")
                continue
            print(countpy,"files in",examples_plainpyomo_dir,"failed")
            return {self.assertEqual(int(flagpy),0),self.assertEqual(int(flagps),0)}
                        
    def test_sawall_examples(self):
        print('##############Sawall###############')
        examples_dir = os.path.join(examplesMainDir, 'sawall')
        self._schedule(examples_dir)

    def test_case51a_examples(self):
        print('##############Case51a##############')
        examples_dir = os.path.join(examplesMainDir, 'case51a')
        self._schedule(examples_dir)

    def test_case51b_examples(self):
        print('##############Case51b##############')
        examples_dir = os.path.join(examplesMainDir, 'case51b')
        self._schedule(examples_dir)

    def test_case51c_examples(self):
        print('##############Case51c##############')
        examples_dir = os.path.join(examplesMainDir, 'case51c')
        self._schedule(examples_dir)

    def test_case51d_examples(self):
        print('##############Case51d##############')
        examples_dir = os.path.join(examplesMainDir, 'case51d')
        self._schedule(examples_dir)

    def test_michaels_examples(self):
        print('##############michaels##############')
        examples_dir = os.path.join(examplesMainDir, 'michaels')
        self._schedule(examples_dir)

    def test_case52a_examples(self):
        print('##############Case52a##############')
        examples_dir = os.path.join(examplesMainDir, 'case52a')
        self._schedule(examples_dir)

    def test_case52b_examples(self):
        print('##############Case52b##############')
        examples_dir = os.path.join(examplesMainDir, 'case52b')
        self._schedule(examples_dir)

    def test_complementary_states_examples(self):
        print('##############complementary_states##############')
        examples_dir = os.path.join(examplesMainDir, 'complementary_states')
        self._schedule(examples_dir)

    def test_plainpyomo_examples(self):
        print('##############plain_pyomo##############')
        examples_dir = os.path.join(examplesMainDir, 'plain_pyomo')
        self._schedule(examples_dir)

    def test_paper_examples(self):
        print('##############paper#############')
        examples_dir = os.path.join(examplesMainDir,'Paper')
        self._schedule(examples_dir)

    def run_all_examples(self):
        self.test_sawall_examples()
        self.test_case51a_examples()
        self.test_case51b_examples()
        self.test_case51c_examples()
        self.test_case51d_examples()
        self.test_michaels_examples()
        self.test_case52a_examples()
        self.test_case52b_examples()
        self.test_complementary_states_examples()
	  self.test_plain_pyomo_examples()
	  self.test_paper_examples()


    def runTest(self):
        print("hallo")  #: XD


if __name__ == '__main__':
    unittest.main()
