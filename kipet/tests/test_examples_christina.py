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
        examples_plainpyomo_dir = os.path.join(examples_dir)
        examples_plainpyomo=[f for f in os.listdir(examples_dir) if os.path.isfile(os.path.join(examples_plainpyomo_dir,f)) and f.endswith('.py')]
        
        if os.path.isdir(os.path.join(examples_dir,'pyomo'))==True:
            #########################
            ##Pyomo-Examples:##
            #########################
            examples_pyomo_dir = os.path.join(examples_dir,'pyomo')
            examples_pyomo = [f for f in os.listdir(examples_pyomo_dir) if os.path.isfile(os.path.join(examples_pyomo_dir,f)) and f.endswith('.py')]
        
            flag = 0
            count = 0
            for f in examples_pyomo:
                #print "running pyomo:",f
                flag = subprocess.call([sys.executable,os.path.join(examples_pyomo_dir,f),'1'],
                                    stdout=self.std_out,
                                    stderr=subprocess.STDOUT)
                if flag!=0:
                    print "running pyomo:",f,"failed"
                    count = count + 1
                    flag=1
                    #outpy=self.assertEqual(int(flag),0)
                else:
                    print "running pyomo:",f,"passed"
                continue
            #########################
            #####Casadi-Examples:####
            #########################
            if found_casadi:
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
                        print "running casadi:",f,"failed"
                        countc = countc + 1
                        flagc=1
                    else:
                        print "running casadi:",f,"passed"
                    continue
                #outcas=self.assertEqual(int(flag),0)
                print countc,"files in",examples_casadi_dir,"failed"
            print count," files in ",examples_pyomo_dir," failed"
            return {self.assertEqual(int(flag),0),self.assertEqual(int(flagc),0)}
        #########################
        ##Plain_Pyomo-Examples:##
        #########################
        else:
            flag=0
            count = 0
            for f in examples_plainpyomo:
                #print "running plainpyomo:",f
                flag = subprocess.call([sys.executable,os.path.join(examples_plainpyomo_dir,f),'1'],
                                    stdout=self.std_out,
                                    stderr=subprocess.STDOUT)
                if flag!=0:
                    print "running plainpyomo:",f,"failed"
                    count = count + 1
                    flag=1
                else:
                    print "running plainpyomo:",f,"passed"
                continue
            print count,"files in",examples_plainpyomo_dir,"failed"
            return self.assertEqual(int(flag),0)
            
                        
    def test_sawall_examples(self):
	print '##############Sawall###############'
        examples_dir = os.path.join(examplesMainDir,'sawall')
        self._schedule(examples_dir)
            
    def test_case51a_examples(self):
	print '##############Case51a##############'
        examples_dir = os.path.join(examplesMainDir,'case51a')
        self._schedule(examples_dir)

    def test_case51b_examples(self):
	print '##############Case51b##############'
        examples_dir = os.path.join(examplesMainDir,'case51b')
        self._schedule(examples_dir)

    def test_case51c_examples(self):
	print '##############Case51c##############'
        examples_dir = os.path.join(examplesMainDir,'case51c')
        self._schedule(examples_dir)

    def test_case51d_examples(self):
	print '##############Case51d##############'
        examples_dir = os.path.join(examplesMainDir,'case51d')
        self._schedule(examples_dir)

    def test_michaels_examples(self):
	print '##############michaels##############'
        examples_dir = os.path.join(examplesMainDir,'michaels')
        self._schedule(examples_dir)

    def test_case52a_examples(self):
	print '##############Case52a##############'
        examples_dir = os.path.join(examplesMainDir,'case52a')
        self._schedule(examples_dir)

    def test_case52b_examples(self):
	print '##############Case52b##############'
        examples_dir = os.path.join(examplesMainDir,'case52b')
        self._schedule(examples_dir)

    def test_complementary_states_examples(self):
	print '##############complementary_states##############'
        examples_dir = os.path.join(examplesMainDir,'complementary_states')
        self._schedule(examples_dir)
        
    def test_plainpyomo_examples(self):
        print '##############plain_pyomo##############'
        examples_dir = os.path.join(examplesMainDir,'plain_pyomo')
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


            
    def runTest(self):
        print "hola"

if __name__ == '__main__':
    unittest.main()
