import unittest
import sys
import os
import inspect
import subprocess

import imp
try:
    imp.find_module('casadi')
    found_casadi=True
except ImportError:
    found_casadi=False

examplesMainDir = os.path.abspath(
        os.path.join( os.path.dirname( os.path.abspath( inspect.getfile(
                    inspect.currentframe() ) ) ), '..','examples'))

class TestExamples(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        sys.path.append(examplesMainDir)
        self.std_out = open("test_examples.log","w")
    @classmethod
    def tearDownClass(self):
        sys.path.remove(examplesMainDir)
        self.std_out.close()

    def _schedule(self,examples_dir):
        examples_pyomo_dir = os.path.join(examples_dir,'pyomo')
        examples_pyomo = [f for f in os.listdir(examples_pyomo_dir) if os.path.isfile(os.path.join(examples_pyomo_dir,f)) and f.endswith('.py')]
        
        flag = 0
        for f in examples_pyomo:
            print("running pyomo:", f)
            flag = subprocess.call([sys.executable,os.path.join(examples_pyomo_dir,f),'1'],
                                   stdout=self.std_out,
                                   stderr=subprocess.STDOUT)
            self.assertEqual(flag,0)
        if found_casadi:
            examples_casadi_dir = os.path.join(examples_dir,'casadi')
            examples_casadi = [f for f in os.listdir(examples_casadi_dir) if os.path.isfile(os.path.join(examples_casadi_dir,f)) and f.endswith('.py')]
            flag = 0
            for f in examples_casadi:
                print("running casadi:", f)
                flag = subprocess.call([sys.executable,os.path.join(examples_casadi_dir,f),'1'],
                                       stdout=self.std_out,
                                       stderr=subprocess.STDOUT)
                self.assertEqual(flag,0)
            
    def test_sawall_examples(self):
        examples_dir = os.path.join(examplesMainDir,'sawall')
        self._schedule(examples_dir)
            
    def test_case51a_examples(self):
        examples_dir = os.path.join(examplesMainDir,'case51a')
        self._schedule(examples_dir)

    def test_case51b_examples(self):
        examples_dir = os.path.join(examplesMainDir,'case51b')
        self._schedule(examples_dir)

    def test_case51c_examples(self):
        examples_dir = os.path.join(examplesMainDir,'case51c')
        self._schedule(examples_dir)

    def run_all_examples(self):
        self.test_sawall_examples()
        self.test_case51a_examples()
        self.test_case51b_examples()
        self.test_case51c_examples()
            
    def runTest(self):
        print("hallo")

if __name__ == '__main__':
    unittest.main()
