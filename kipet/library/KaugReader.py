#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 17:52:54 2020

@author: kevin
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import shutil

from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

list_of_k_aug_files = [ 'a0',
                        'a1',
                        'anz_p_row.in',
                        'conorder.txt',
                        'dot_in_.in',
                        'dsdp_in_.in',
                        'dxdp_.dat',
                        'grad_debug_sorted.in',
                        'hess_debug.in',
                        'inv_.in',
                        'inv_red_hess',
                        'ipopt.opt',
                        'jacobi_debug.in',
                        'k_aug_hess',
                        'kkt.in',
                        'mc19_results.txt',
                        'md_positions.in',
                        'my_ouput.txt',
                        'nz_per_row.in',
                        'primal0.txt',
                        'primal1.txt',
                        'primal_dual.txt',
                        'problem.nl',
                        'result_lin_sol.txt',
                        'result_primal_dual.txt',
                        'result_red_hess.txt',
                        'result_unscaled.txt',
                        'rhs_sens_scaled',
                        'rhs_dcdp',
                        'row_start.in',
                        'scale_facts.txt',
                        'sigma_out',
                        'sigma_super_basic.txt',
                        'sigma_warnings.txt',
                        'timings_k_aug.txt',
                        'timings_k_aug_dsdp.txt',
                        'varorder.txt',
                        'zx.txt',
                      ]

class KaugReader():
    """Class to read k_aug files and convert them to useful formats"""
    
    
    def __init__(self, 
                 name=None, 
                 file=None, 
                 pyomo_file=None, 
                 clean=False,
                 verbose=False,
                 read_only=False,
                 ):
        
        if file is None:
            self.file = Path.cwd()
        else:
            self.file = Path(file)
        
        if name is None:
            self.name = 'scenario'
        else:
            self.name = name
        
        if pyomo_file is not None:
            self.pyomo_file = pyomo_file
            self.use_model_output = True
        else:
            self.pyomo_file = None
            self.use_model_output = False
            
        self.verbose = verbose
        self.clean = clean
        
        self.core_file_directory = Path.cwd().joinpath('k_aug_output')
        self.file_directory = self.core_file_directory.joinpath(self.name)
        
        self.read_only = read_only
        if not read_only:
            self._make_directory()
            self._move_pyomo_results()
        else:
            self._move_files()
        
        self.dimensions = self._read_problem_dimensions()
        
    def _make_directory(self):
        """Creates directories and moves the k_aug files to the newly created
        directory
        
        Returns:
            
            None
        
        """        
        if not self.core_file_directory.is_dir():
            self.core_file_directory.mkdir(exist_ok=True)
            if self.clean == True:
                self._nuclear_option()
             
        counter = 1
        
        try:
            self.file_directory.mkdir(exist_ok=False)
        except:
                  
            while True:
                  
                self.file_directory = self.core_file_directory.joinpath(self.name + f'_{counter}')
                
                try:
                    self.file_directory.mkdir(exist_ok=False)
                    print(f'This directory >> {self.file_directory} << already exists!')
                    print(f'Renaming to {self.file_directory.name[:-len(str(counter))]+str(counter+1)}')
                    break
                except:
                    counter += 1

                if counter == 10000:
                    raise ValueError('Man, get a grip on the naming convention!')
                    break
                
        self._move_files()
            
    def _move_files(self):
        """Move the files to the designated folder"""
        
        if self.pyomo_file is not None and not self.read_only:
            list_of_k_aug_files.append(self.pyomo_file)
        
        for file in list_of_k_aug_files:
            if Path(file).is_file():    
                if self.verbose:
                    print(f'Moving {file} to {self.file_directory}')
                shutil.move(Path.cwd().joinpath(file), self.file_directory.joinpath(file))
            
        return None
    
    def _read_problem_dimensions(self):
        """This reads the jacobi_debug.in file for the first line to get the
        dimensions of the problem which are used in all functions
        
        Returns:
            
            dimensions (dict): dict with m and n dimensions of the problem
        """
        file = Path('jacobi_debug.in')
        with open(self.file_directory.joinpath(file), 'r') as file:
            dimensions = file.readline()
        
        dimensions = [int(d.rstrip()) for d in dimensions.split('\t')]
        
        dimensions = {'n' : dimensions[1],
                      'm' : dimensions[0],
                      }
        
        return dimensions
    
    def _read_kkt(self, dense=True):
        """Function for reading the kkt.in file and building the full KKT-
        Matrix.
        
        Args:
            
            dense (bool, optional): return a dense matrix
            
        Returns:
            
            kkt (np.ndarray, coo_matrix): kkt matrix in either sparse or
            dense form
        
        """
        file = self.file_directory.joinpath('kkt.in')
        
        if file.is_file():
             
            A_transposed = self._read_jacobian()
    
            mat = pd.read_csv(self.file_directory.joinpath(file), 
                              delim_whitespace=True, 
                              header=None, 
                              skipinitialspace=True
                              )
            mat.columns = ['irow', 'jcol', 'vals']
            mat.irow -= 1
            mat.jcol -= 1
            
            m = self.dimensions['m']
            n = self.dimensions['n']
            mn = m + n 
    
            kkt_coo = coo_matrix((mat.vals, (mat.irow, mat.jcol)), shape=(mn, mn)) 
            kkt = np.asarray(kkt_coo.todense())
            
            J = np.asarray(A_transposed)#
            J = J.reshape(self.dimensions['m'], self.dimensions['n'])
            bottom_matrices = np.hstack((J, np.zeros((self.dimensions['m'], self.dimensions['m']))))
    
            kkt[n:, :] = bottom_matrices
            
            if not dense:
                return coo_matrix(kkt)
            else:
                return kkt
       
        else:
            print(f'File {file.name} does not exist.')
            return None
        
    def _read_hessian(self, dense=True):
        """Function for reading the hess_debug.in file
        
        Args:
            
            dense (bool, optional): return a dense matrix
            
        Returns:
            
            hess_coo (np.ndarray, coo_matrix): kkt matrix in either sparse or
            dense form
        
        """
        file = Path('hess_debug.in')
        hess = pd.read_csv(self.file_directory.joinpath(file), 
                           delim_whitespace=True, 
                           header=None, 
                           skipinitialspace=True
                           )
        hess.columns = ['irow', 'jcol', 'vals']
        hess.irow -= 1
        hess.jcol -= 1
        
        n = self.dimensions['n']
        hess_coo = coo_matrix((hess.vals, (hess.irow, hess.jcol)), shape=(n, n)) 
        
        if not dense:
            return hess_coo
        else:
            return hess_coo.todense()
        
    def _read_jacobian(self, dense=True):
        """Function for reading the jacobi_debug.in file - This is actually the
        hessian of the contraints
        
        Args:
            
            dense (bool, optional): return a dense matrix
            
        Returns:
            
            jac_coo (np.ndarray, coo_matrix): Jacobian matrix in either sparse 
            or dense form
        
        """
        file = Path('jacobi_debug.in')
        jac = pd.read_csv(self.file_directory.joinpath(file),
                          delim_whitespace=True,
                          header=None,
                          skipinitialspace=True
                          )
        m = jac.iloc[0,0]
        n = jac.iloc[0,1]
        jac.drop(index=[0], inplace=True)
        jac.columns = ['irow', 'jcol', 'vals']
        jac.irow -= 1
        jac.jcol -= 1
        
        m = self.dimensions['m']
        n = self.dimensions['n']
        jac_coo = coo_matrix((jac.vals, (jac.irow, jac.jcol)), shape=(m, n)) 
        
        if not dense:
            return jac_coo
        else:
            return jac_coo.todense()
        
    def _read_rhs_scaled(self):
        
        rhs_file = self.file_directory.joinpath('rhs_sens_scaled')
        
        return pd.read_csv(rhs_file,
                           delim_whitespace=True,
                           skipinitialspace=True,
                           header=None) # dummy sep
        #self.col_ind = [var_ind.loc[var_ind[0] == f'P[{v}]'].index[0] for v in Se]
    
    def _read_rhs_dcdp(self):
        
        rhs_file = self.file_directory.joinpath('rhs_dcdp')
        
        return pd.read_csv(rhs_file,
                           delim_whitespace=True,
                           skipinitialspace=True,
                           #header=None,
                           ) # dummy sep
        
    def _read_dxdp(self):
        
        dxdp_file = self.file_directory.joinpath('dxdp_.dat')
        
        return pd.read_csv(dxdp_file,
                           delim_whitespace=True,
                           skipinitialspace=True,
                           header=None,
                           ) # dummy sep
        
    
    def _read_var_list(self):
        
        col_file = self.file_directory.joinpath('pyomo_results.col')
        
        return pd.read_csv(col_file, sep = ';', header=None) # dummy sep
        
    def _read_con_list(self):
        
        row_file = self.file_directory.joinpath('pyomo_results.row')
        df_con = pd.read_csv(row_file, sep = ';', header=None) # dummy sep
        
        if df_con.iloc[-1].values[0] == 'objective':
            df_con_no_objective = df_con.drop(index=[df_con.index.stop - 1])
        
            return df_con_no_objective
        
        else:
            raise ValueError('There may be an issue with the objective in the jacobian')
        
    def _kkt_dataframe(self):
        
        df_index = pd.concat((self.variable_index, self.constraint_index))
        labels = list(df_index[0])
        
        kkt = pd.DataFrame(self.kkt, columns=labels, index=labels)
        
        return kkt
    
        # Jac_f = Jac[:, col_ind]
        # Jac_l = np.delete(Jac, col_ind, axis=1)
        # X = spsolve(coo_matrix(np.mat(Jac_l)).tocsc(), coo_matrix(np.mat(-Jac_f)).tocsc())
        
        # col_ind_left = list(set(range(n)).difference(set(col_ind)))
        # col_ind_left.sort()
        
        # Z = np.zeros([n, n_free])
        # Z[col_ind, :] = np.eye(n_free)
        # Z[col_ind_left, :] = X.todense()
    
        # Z_mat = coo_matrix(np.mat(Z)).tocsr()
        # Z_mat_T = coo_matrix(np.mat(Z).T).tocsr()
        # Hess = Hess_coo.tocsr()
        # red_hessian = Z_mat_T * Hess * Z_mat
        
        return None
        
    def _move_pyomo_results(self):
        """This moves the solver results from the /tmp directory to the current
        directory with the k_aug files.
        

        Returns
        -------
        None.

        """
        if not self.use_model_output:
            print('You need to specify the log file for the model results')
        
            return None
        
        else:
        
            pyomo_file = self.file_directory.joinpath(self.pyomo_file)
            with open(pyomo_file, 'r') as f:
                output_string = f.read()
            
            stub = output_string.split('\n')[0].split(',')[1][2:-4]
            output_file_ext = ['.col', '.sol', '.row', '.nl']
            
            for ext in output_file_ext:
            
                file = Path(stub + ext)
                moved_file = self.file_directory.joinpath(file.name)
                shutil.move(str(file), str(moved_file))
                dirpath = moved_file.parent
                suffix = moved_file.suffix
                filename = 'pyomo_results' + suffix
                filepath = dirpath / filename
                moved_file.rename(filepath)
                if self.verbose:
                    print(f'Moving {file} to {filepath}')
                
            return None
                
        
    def clear_all_files(self):
        """Removes all of the k_aug files from the current directory
        
        Args:
            
            directory (string, Path): path to the directory where the files
                were generated
        
        Return:
            None
        
        """
        shutil.rmtree(self.file_directory)
        print(f'Deleting directory: {self.file_directory}')
            
        return None
        
    def _nuclear_option(self):
        """Deletes all subdirectories in file_directory! Take care!"""
        
        print('You have nuked all of your files.')
        for file in Path('./k_aug_output/').iterdir():
            if file.is_dir():
                shutil.rmtree(file)
    
        return None # literally
    
    @property
    def dxdp(self):
        return self._read_dxdp()
    
    @property
    def rhs_sens_scaled(self):
        return self._read_rhs_scaled()

    @property
    def rhs_dcdp(self):
        return self._read_rhs_dcdp()

    @property
    def variable_index(self):
        return self._read_var_list()
    
    @property
    def constraint_index(self):
        return self._read_con_list()
    
    @property
    def kkt_df(self):
        return self._kkt_dataframe()
    
    @property
    def kkt(self):
        return self._read_kkt(dense=True)
    
    @property
    def kkt_sparse(self):
        return self._read_kkt(dense=False)
    
    @property
    def jacobian(self):
        return self._read_jacobian(dense=True)
    
    @property
    def jacobian_sparse(self):
        return self._read_jacobian(dense=False)
    
    @property
    def hessian(self):
        return self._read_hessian(dense=True)
    
    @property
    def hessian_sparse(self):
        return self._read_hessian(dense=False)
    
def move_pyomo_results(tmpfile, directory, verbose=True):
        """This moves the solver results from the /tmp directory to the current
        directory with the k_aug files.
        

        Returns
        -------
        None.

        """
        
        core_file_directory = Path.cwd().joinpath('k_aug_output')
        file_directory = core_file_directory.joinpath(directory)
        file_directory.mkdir(exist_ok=True)
        
        pyomo_file = Path.cwd().joinpath(Path(tmpfile)) 
     
        with open(pyomo_file, 'r') as f:
            output_string = f.read()
        
        stub = output_string.split('\n')[0].split(',')[1][2:-4]
        output_file_ext = ['.col', '.sol', '.row', '.nl']
        
        for ext in output_file_ext:
        
            file = Path(stub + ext)
            moved_file = Path(file_directory).joinpath(file.name)
            shutil.move(str(file), str(moved_file))
            dirpath = moved_file.parent
            suffix = moved_file.suffix
            filename = 'pyomo_results' + suffix
            filepath = dirpath / filename
            moved_file.rename(filepath)
            
            if verbose:
                print(f'Moving {file} to {filepath}')
            
        return None
    
    
if __name__ == '__main__':
    pass




