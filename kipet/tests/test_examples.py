import unittest
import sys
import os
import inspect
import subprocess
examplesMainDir = os.path.abspath(
        os.path.join( os.path.dirname( os.path.abspath( inspect.getfile(
                    inspect.currentframe() ) ) ), '..','examples'))

class TestExamples(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        sys.path.append(examplesMainDir)
        sys.path.append(os.path.join(examplesMainDir,'sawall'))

    @classmethod
    def tearDownClass(self):
        sys.path.remove(examplesMainDir)

    def test_sawall_examples(self):
        example_files = [f for f in os.listdir(os.path.join(examplesMainDir,'sawall')) if os.path.isfile(os.path.join(examplesMainDir,'sawall',f)) and f.endswith('.py')]
        
        #print all_files
        flag = 0
        for f in example_files:
            print "running:",f
            flag = subprocess.call([sys.executable,os.path.join(examplesMainDir,'sawall',f),'1'])
            self.assertEqual(flag,0)

    def test_case51a_examples(self):
        example_files = [f for f in os.listdir(os.path.join(examplesMainDir,'case51a')) if os.path.isfile(os.path.join(examplesMainDir,'case51b',f)) and f.endswith('.py')]
        
        #print all_files
        flag = 0
        for f in example_files:
            print "running:",f
            flag = subprocess.call([sys.executable,os.path.join(examplesMainDir,'case51a',f),'1'])
            self.assertEqual(flag,0)

    def test_case51b_examples(self):
        example_files = [f for f in os.listdir(os.path.join(examplesMainDir,'case51b')) if os.path.isfile(os.path.join(examplesMainDir,'case51b',f)) and f.endswith('.py')]
        
        #print all_files
        print example_files
        flag = 0
        for f in example_files:
            print "running:",f
            flag = subprocess.call([sys.executable,os.path.join(examplesMainDir,'case51b',f),'1'])
            self.assertEqual(flag,0)

    def test_case51c_examples(self):
        example_files = [f for f in os.listdir(os.path.join(examplesMainDir,'case51c')) if os.path.isfile(os.path.join(examplesMainDir,'case51c',f)) and f.endswith('.py')]
        flag = 0
        for f in example_files:
            print "running:",f
            flag = subprocess.call([sys.executable,os.path.join(examplesMainDir,'case51c',f),'1'])
            self.assertEqual(flag,0)

    def run_all_examples(self):
        self.test_sawall_examples()
        self.test_case51a_examples()
        self.test_case51b_examples()
        self.test_case51c_examples()
            
    def runTest(self):
        print "hola"

if __name__ == '__main__':
    unittest.main()
