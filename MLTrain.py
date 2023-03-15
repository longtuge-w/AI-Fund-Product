#!/usr/bin/env python
#-*- encoding:utf-8 -*-
"""
Created on March 24 2019
@author: yout
@email: yout@bosera.com
"""
import os,datetime,functools
import pickle
import deepforest
import numpy as np
import pandas as pd
import xgboost
from modelMetric import runningMetric
# from sklearn.externals import joblib
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier,RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.linear_model import LogisticRegression,SGDClassifier,Ridge,Lasso,ElasticNet,RidgeClassifier
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR,SVC,LinearSVC
# from sklearn import svm
# from sklearn.svm import LinearSVC
from xgboost import XGBClassifier,XGBRegressor
from lightgbm import LGBMClassifier,LGBMRegressor
from catboost import CatBoostClassifier,CatBoostRegressor
from deepforest import CascadeForestClassifier,CascadeForestRegressor
#from sklearn.cross_validation import train_test_split	#for sklearn version under 0.19. 
from sklearn.model_selection import train_test_split	#for sklearn version above 0.20. 
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from hyperopt import fmin,tpe,hp
from hyperopt.pyll.base import scope
from pydash import at
from sklearn.utils import check_array


__all__ = ['fileAccess','staticMethod','trailingMethod','cyclicalTrailMethod']

# 17 models update: Regression(kernelRidge/ridge/lasso/elasticNet/svr);
# Classification(randomForest/gbdt/mlpc/gnb/adaboost/xgboost/lightgbm/catboost/logistic/sgd/linearSVC/svm)

# used to calculate the running time of one specific model
def funcTimer(func):
	@functools.wraps(func)
	def wrapper(*args,**kwargs):
		t1 = datetime.datetime.now()
		ret = func(*args,**kwargs) 
		t2 = datetime.datetime.now()
		print('%s run time:%s'%(func.__name__,(t2-t1)))
		return ret 
	return wrapper


# used to calculate the running time of training and storing one special model by trailing or cyclical method
def timeLog(text1,text2):
	def funcTimer(func):
		@functools.wraps(func)
		def wrapper(*args,**kwargs):
			t1 = datetime.datetime.now()
			ret = func(*args,**kwargs) 
			t2 = datetime.datetime.now()
			print('%s_%s_%s run time:%s'%(text1,text2,func.__name__,(t2-t1)))
			return ret 
		return wrapper
	return funcTimer


# def saveLog(text1,text2,text3):
# 	def saveInfo(func):
# 		@functools.wraps(func)
# 		def wrapper(*args,**kwargs):
# 			ret = func(*args,**kwargs)
# 			print('%s_%s_%s %s saved in %s'%(text1,func.__name__,text2,text3,ret))
# 			return ret 
# 		return wrapper
# 	return saveInfo

# def rankICLog(text1,text2):
# 	def rankIC(func):
# 		@functools.wraps(func)
# 		def wrapper(*args,**kwargs):
# 			ret = func(*args,**kwargs)

# 			return ret 



# used to read or store data and model
class fileAccess(object):
	'''
	default file path contain data model results
	'''
	__dataPath = 'DataHouse/'
	__rawDataPath = 'rawData/' 
	__processedDataPath = 'processedData/'
	__modelPath = 'ModelHouse/'
	__resultPath = 'Results/'

	def getPath(self):
		return {'__dataPath':self.__dataPath,
				'__rawDataPath':self.__rawDataPath,
				'__processedDataPath':self.__processedDataPath,
				'__modelPath':self.__modelPath,
				'__resultPath':self.__resultPath}
	
	def setDataPath(self,dataPath):
		self.__dataPath = dataPath

	def setRawDataPath(self,rawDataPath):
		self.__rawDataPath = rawDataPath

	def setProcessedDataPath(self,processedDataPath):
		self.__processedDataPath = processedDataPath

	def setModelPath(self,modelPath):
		self.__modelPath = modelPath

	def setResultPath(self,resultPath):
		self.__resultPath = resultPath

    # generate path for building model
	def modelPathGenerator(self,trainMethod,modelName,paraTuneMethod,ortho_method,*args):
		if ortho_method == 'none':
			path = self.__modelPath + trainMethod + '/' + modelName + '/' + paraTuneMethod + '/'
		else:
			path = self.__modelPath + trainMethod + '/' + modelName + '_' + ortho_method + '/' + paraTuneMethod + '/'
		if not os.path.exists(path):
			os.makedirs(path)
		return path + '/'.join(args) 
		
    # storing model under the generated path
	def modelStorer(self,trainedModel,trainMethod,modelName,paraTuneMethod,ortho_method,*args):
		path = self.modelPathGenerator(trainMethod,modelName,paraTuneMethod,ortho_method,*args)
		joblib.dump(trainedModel,path)
		print('%s_%s_%s trained model saved in %s'%(trainMethod,modelName,paraTuneMethod,path))
		print(' '*10)


class Splits(object):
	def __init__(self) -> None:
		super().__init__()


	def cross_validation(self,model,X,y,params,mission):
		cv = params.get('cv',4)
		if mission == 'classification':
			scoring = params['cross_val_score'].get('cross_val_score','roc_auc')
		elif mission == 'regression':
			scoring = params['cross_val_score'].get('cross_val_score','neg_mean_squared_error')
		print('use cross validation with parameters cv=%s, scoring=%s'%(cv,scoring))
		return np.nanmean(cross_val_score(model,X,y,cv=cv,scoring=scoring))


	def time_series_validation(self,model,X,y,params):
		n_splits = params['time_series_score'].get('n_splits',5)
		test_size = params['time_series_score'].get('test_size',None)
		score_lst = []
		tscv = TimeSeriesSplit(n_splits=n_splits,test_size=test_size)
		print('use time series split validation with parameters n_splits=%s, test_size=%s'%(n_splits,test_size))
		trainX,trainY = X.values,y.values
		for train_index, test_index in tscv.split(trainX):
			train_X,test_X = trainX[train_index],trainX[test_index]
			train_Y,test_Y = trainY[train_index],trainY[test_index]
			modelTrained = model.fit(train_X,train_Y)
			score = modelTrained.score(test_X,test_Y)
			score_lst.append(score)
		return np.nanmean(score_lst)


	def group_time_series_validation(self,model,X,y,groups,params):
		n_splits = params['group_time_series_score'].get('n_splits',5)
		score_lst = []
		trainX,trainY,groups = X.values,y.values,groups.values

		if groups is None:
			raise ValueError("The 'groups' parameter should not be None.")
		groups = check_array(groups, ensure_2d=False, dtype=None)

		unique_groups = np.unique(groups, return_inverse=True)[0]
		n_groups = len(unique_groups)

		if n_groups%n_splits!=0:
			raise ValueError("Cannot have number of splits n_splits=%d that cannot be divided by"
								" the number of groups: %d."
								% (n_splits, n_groups))

		groups = np.array(groups)
		n_groups_per_fold = int(n_groups/n_splits)

		for n_split in range(n_splits-1):
			train_groups = unique_groups[:n_groups_per_fold*(n_split+1)]
			test_groups = unique_groups[n_groups_per_fold*(n_split+1):n_groups_per_fold*(n_split+2)]
			train_index = np.where(np.logical_and(groups>=train_groups[0],groups<=train_groups[-1]))[0]
			test_index = np.where(np.logical_and(groups>=test_groups[0],groups<=test_groups[-1]))[0]
			train_X,test_X = trainX[train_index],trainX[test_index]
			train_Y,test_Y = trainY[train_index],trainY[test_index]
			modelTrained = model.fit(train_X,train_Y)
			score = modelTrained.score(test_X,test_Y)
			score_lst.append(score)

		print('use group time series split validation with parameters n_split=%s'%(n_splits))

		return np.nanmean(score_lst)


# used to select model and mission to train model
class modelSelector(object):
	'''
	3 train method to select: default parameter / user set parameter / optimize parameter by gridSearchCV/hyperopt/...
	2 hyperparameter optimization method: grid search and bayesian optimization
	preset GridSearchCV with cv=4 / n_job=-1 
	17 models update: 
	Regression(kernelRidge/ridge/lasso/elasticNet/svr/randomForest/gbdt/mlpc/adaboost/xgboost/lightgbm/catboost/sgd);
	Classification(ridge/randomForest/gbdt/mlpc/gnb/adaboost/xgboost/lightgbm/catboost/logistic/sgd/linearSVC/svm)
	fast model: kernelRidge/ridge/lasso/elasticNet/svr/gnb/sgd/linearSVC/svm
	slow model: randomForest/gbdt/mlpc/adaboost/xgboost/lightgbm/catboost
	'''
	def __init__(self,paraTuneMethod,trainF,trainL,trainR,testF,testL,testR,trainTimeID,trainStockID,validation=()):
		self.trainF = trainF # X
		self.trainL = trainL # Y, used for classification
		self.trainR = trainR # Y, used for regression 
		self.testF = testF
		self.testL = testL
		self.testR = testR
		self.paraTuneMethod = paraTuneMethod
		self.trainTimeID = trainTimeID
		self.trainStockID = trainStockID
		if len(validation) != 0:
			self.validation = validation[0]
		else:
			self.validation = None
		# uniform: [min,max], return a float number between min and max randomly
		# int: [min,max], return a integer between min and max randomly
		# choice: [choice 1, choice 2, ... , choice n], return one of the choice randomly
		self.model_params = {
			'kernelridge':{
				'uniform':['alpha','gamma','coef0'],
				'int':['degree'],
				'choice':['kernel'],
			},
			'ridge':{
				'regression':{
					'uniform':['alpha','tol'],
					'int':['max_iter'],
					'choice':['fit_intercept','normalize','copy_X','solver','random_state']
				},
				'classification':{
					'uniform':['alpha','tol'],
					'int':['max_iter'],
					'choice':['fit_intercept','normalize','copy_X','solver','random_state']
				}				
			},
			'lasso':{
				'uniform':['alpha','tol'],
				'int':['max_iter'],
				'choice':['fit_intercept','normalize','copy_X','solver','random_state']
			},
			'elasticnet':{
				'uniform':['alpha','tol','l1_ratio'],
				'int':['max_iter','random_state'],
				'choice':['fit_intercept','normalize','copy_X','random_state']
			},
			'svr':{
				'uniform':['coef0','tol','C','epsilon','cache_size','gamma'],
				'int':['max_iter','degree'],
				'choice':['kernel','shrinking','verbose']
			},
			'randomforest':{
				'regression':{
					'uniform':['min_weight_fraction_leaf','min_impurity_decrease','min_impurity_split','ccp_alpha'],
					'int':['min_samples_split','n_estimators','max_depth','max_leaf_nodes','n_jobs','random_state','verbose','min_sample_leaf','max_features','max_samples'],
					'choice':['criterion','bootstrap','oob_score','warm_start']
				},
				'classification':{
					'uniform':['min_weight_fraction_leaf','min_impurity_decrease','min_impurity_split','ccp_alpha'],
					'int':['min_samples_split','n_estimators','max_depth','max_leaf_nodes','n_jobs','random_state','verbose','min_sample_leaf','max_features','max_samples'],
					'choice':['criterion','bootstrap','oob_score','warm_start','class_weight']
				}				
			},
			'gbdt':{
				'regression':{
					'uniform':['subsample','min_weight_fraction_leaf','max_features','min_impurity_decrease','min_impurity_split','ccp_alpha','max_samples','validation_fraction','tol','alpha'],
					'int':['n_estimators','max_depth','max_leaf_nodes','random_state','verbose','n_iter_no_change','min_samples_split','min_samples_leaf'],
					'choice':['loss','criterion','warm_start']
				},
				'classification':{
					'uniform':['subsample','min_weight_fraction_leaf','max_features','min_impurity_decrease','min_impurity_split','ccp_alpha','max_samples','validation_fraction','tol','alpha'],
					'int':['n_estimators','max_depth','max_leaf_nodes','random_state','verbose','n_iter_no_change','min_samples_split','min_samples_leaf'],
					'choice':['loss','criterion','warm_start']
				}				
			},
			'mlpc':{
				'regression':{
					'uniform':['alpha','power_t','tol','momentum','validation_fraction','beta_1','beta_2','epsilon'],
					'int':['batch_size','max_iter','random_state','n_iter_no_change','max_fun'],
					'choice':['hidden_layer_sizes','activation','solver','learning_rate_init','shuffle','verbose','warm_start','nesterovs_momentum','early_stopping']
				},
				'classification':{
					'uniform':['alpha','power_t','tol','momentum','validation_fraction','beta_1','beta_2','epsilon'],
					'int':['batch_size','max_iter','random_state','n_iter_no_change','max_fun'],
					'choice':['hidden_layer_sizes','activation','solver','learning_rate_init','shuffle','verbose','warm_start','nesterovs_momentum','early_stopping']
				}				
			},
			'gnb':{
				'uniform':['var_smoothing'],
				'int':[],
				'choice':['priors']
			},
			'adaboost':{
				'regression':{
					'uniform':['learning_rate'],
					'int':['n_estimators','random_state'],
					'choice':['base_estimator','loss']
				},
				'classification':{
					'uniform':['learning_rate'],
					'int':['n_estimators','random_state'],
					'choice':['base_estimator','algorithm']
				}				
			},
			'xgboost':{
				'regression':{
					'uniform':['learning_rate','gamma','min_child_weight','max_delta_step','subsample','colsample_bytree','colsample_bylevel','colsample_bynode','reg_alpha','reg_lambda','scale_pos_weight','base_score'],
					'int':['n_estimators','max_depth','verbosity','n_jobs'],
					'choice':['objective','booster','tree_method','random_state']
				},
				'classification':{
					'uniform':['learning_rate','gamma','min_child_weight','max_delta_step','subsample','colsample_bytree','colsample_bylevel','colsample_bynode','reg_alpha','reg_lambda','scale_pos_weight','base_score'],
					'int':['n_estimators','max_depth','verbosity','n_jobs'],
					'choice':['use_label_encoder','objective','booster','tree_method','random_state']
				}				
			},
			'lightgbm':{
				'regression':{
					'uniform':['learning_rate','min_split_gain','min_child_weight','min_child_samples','subsample','colsample_bytree','reg_alpha','reg_lambda'],
					'int':['num_leaves','max_depth','n_estimators','subsample_for_bin','subsample_freq','random_state','n_jobs'],
					'choice':['boosting_type','objective','class_weight','silent','importance_type']
				},
				'classification':{
					'uniform':['learning_rate','min_split_gain','min_child_weight','min_child_samples','subsample','colsample_bytree','reg_alpha','reg_lambda'],
					'int':['num_leaves','max_depth','n_estimators','subsample_for_bin','subsample_freq','random_state','n_jobs'],
					'choice':['boosting_type','objective','class_weight','silent','importance_type']
				}				
			},
			'catboost':{
				'regression':{
					'uniform':['learning_rate','l2_leaf_reg','bagging_temperature','subsample','mvs_reg','random_strength','rsm','fold_len_multiplier','scale_pos_weight','diffusion_temperature','penalties_coefficient','model_shrink_rate',
					'od_pval','target_border','gpu_ram_part','model_size_reg'],
					'int':['iterations','random_seed','best_model_min_trees','depth','min_data_in_leaf','max_leaves','one_hot_max_size','fold_permutation_block','leaf_estimation_iterations','early_stopping_rounds','od_wait',
					'border_count','classes_count','thread_count','used_ram_limit','pinned_memory_size','metric_period','snapshot_interval','ctr_target_border_count','max_ctr_complexity',
					'ctr_leaf_count_limit','ctr_leaf_count_limit'],
					'choice':['loss_function','custom_metric','eval_metric','bootstrap_type','sampling_frequency','sampling_unit','use_best_model','grow_policy','ignored_features','has_time','nan_mode','input_borders','output_borders',
					'leaf_estimation_method','leaf_estimation_backtracking','approx_on_full_history','class_weights','class_names','auto_class_weights','boosting_type','boost_from_average','langevin','posterior_sampling','allow_const_label',
					'score_function','monotone_constraints','feature_weights','first_feature_use_penalties','per_object_feature_penalties','model_shrink_mode','tokenizers','dictionaries','feature_calcers','text_processing','cd_type',
					'feature_border_type','per_float_feature_quantization','gpu_cat_features_storage','data_partition','task_type','devices','name','logging_level','verbose','train_final_model','train_dir','allow_writing_files','save_snapshot',
					'snapshot_file','roc_file','simple_ctr','combinations_ctr','per_feature_ctr','counter_calc_method','store_all_simple_ctr','final_ctr_computation_mode']
				},
				'classification':{
					'uniform':['learning_rate','l2_leaf_reg','bagging_temperature','subsample','mvs_reg','random_strength','rsm','fold_len_multiplier','scale_pos_weight','diffusion_temperature','penalties_coefficient','model_shrink_rate',
					'od_pval','target_border','gpu_ram_part','model_size_reg'],
					'int':['iterations','random_seed','best_model_min_trees','depth','min_data_in_leaf','max_leaves','one_hot_max_size','fold_permutation_block','leaf_estimation_iterations','early_stopping_rounds','od_wait',
					'border_count','classes_count','thread_count','used_ram_limit','pinned_memory_size','metric_period','snapshot_interval','ctr_target_border_count','max_ctr_complexity',
					'ctr_leaf_count_limit','ctr_leaf_count_limit'],
					'choice':['loss_function','custom_metric','eval_metric','bootstrap_type','sampling_frequency','sampling_unit','use_best_model','grow_policy','ignored_features','has_time','nan_mode','input_borders','output_borders',
					'leaf_estimation_method','leaf_estimation_backtracking','approx_on_full_history','class_weights','class_names','auto_class_weights','boosting_type','boost_from_average','langevin','posterior_sampling','allow_const_label',
					'score_function','monotone_constraints','feature_weights','first_feature_use_penalties','per_object_feature_penalties','model_shrink_mode','tokenizers','dictionaries','feature_calcers','text_processing','cd_type',
					'feature_border_type','per_float_feature_quantization','gpu_cat_features_storage','data_partition','task_type','devices','name','logging_level','verbose','train_final_model','train_dir','allow_writing_files','save_snapshot',
					'snapshot_file','roc_file','simple_ctr','combinations_ctr','per_feature_ctr','counter_calc_method','store_all_simple_ctr','final_ctr_computation_mode']
				}				
			},
			'logistic':{
				'uniform':['tol','C','intercept_scaling','l1_ratio'],
				'int':['random_state','max_iter','verbose','n_jobs'],
				'choice':['penalty','dual','fit_intercept','class_weight','solver','multi_class','warm_start']
			},
			'sgd':{
				'regression':{
					'uniform':['alpha','l1_ratio','tol','epsilon','eta0','power_t','validation_fraction'],
					'int':['max_iter','verbose','random_state','n_iter_no_change'],
					'choice':['loss','penalty','fit_intercept','shuffle','learning_rate','early_stopping','class_weight','warm_start','average']
				},
				'classification':{
					'uniform':['alpha','l1_ratio','tol','epsilon','eta0','power_t','validation_fraction'],
					'int':['max_iter','verbose','n_jobs','random_state','n_iter_no_change'],
					'choice':['loss','penalty','fit_intercept','shuffle','learning_rate','early_stopping','class_weight','warm_start','average']
				}				
			},
			'linearsvc':{
				'uniform':['tol','C','intercept_scaling'],
				'int':['verbose','random_state','max_iter'],
				'choice':['penalty','loss','dual','multi_class','fit_intercept','class_weight']
			},
			'svm':{
				'uniform':['C','coef0','tol','cache_size'],
				'int':['degree','max_iter','random_state'],
				'choice':['kernel','gamma','shrinking','probability','class_weight','verbose','decision_function_shape','break_ties']
			},
			'deepforest':{
				'regression':{
					'uniform':['delta'],
					'int':['n_bins','bin_subsample','max_layers','n_estimators','n_trees','max_depth','min_samples_split','min_sample_leaf','n_tolerant_rounds','n_jobs','random_state','verbose'],
					'choice':['bin_type','criterion','use_predictor','predictor','predictor_kwargs','backend','partial_mode']
				},
				'classification':{
					'uniform':['delta'],
					'int':['n_bins','bin_subsample','max_layers','n_estimators','n_trees','max_depth','min_samples_split','min_sample_leaf','n_tolerant_rounds','n_jobs','random_state','verbose'],
					'choice':['bin_type','criterion','use_predictor','predictor','predictor_kwargs','backend','partial_mode']
				}
			}
		}


    # kernel ridge regression
	def kernelRidge(self,mission='regression',**kwargs):
		if mission == 'regression':
			# No optimization for hyperparameter, but user can still pass parameter
			if self.paraTuneMethod == 'NoHyperTune':
				print('No hyperparameter optimization needed, now training the model')
				kernelRidge = KernelRidge(**kwargs)
				kernelRidgeTrained = kernelRidge.fit(self.trainF,self.trainR)

			# grid search method
			elif self.paraTuneMethod == 'gridsearchcv':
				print('use grid search method to do hyperparameter optimizaton, now training the model')
				kernelRidge = KernelRidge()	
				
				# default setting
				if len(kwargs) == 0:
					paramSpace = {'alpha': [1e0, 0.1, 1e-2, 1e-3], # coefficient for regularization term
								'gamma': np.logspace(-2, 2, 5), # coefficient for kernel function
								'kernel':'rbf' # kernel function
								}
				else:
					paramSpace = kwargs
				# grid search method, cv：cross-validation generator，n_job：the number of parallel jobs
				kernelRidge = GridSearchCV(kernelRidge,paramSpace,cv=4,n_jobs=-1)
				kernelRidgeTrained = kernelRidge.fit(self.trainF,self.trainR)

			# bayesian optimization method
			elif self.paraTuneMethod == 'hyperopt':
				print('use bayesian hyperparameter optimization, now training the model')

				def kernelridge(paramSpace):
					kernelridge = KernelRidge()
					kernelridge.set_params(**paramSpace)
					if self.validation == None:
						score = cross_val_score(kernelridge,self.trainF,self.trainR,cv=4,scoring='neg_mean_squared_error').mean()
						print('use cross validation with parameters cv=4, scoring=neg_mean_squared_error')
					elif self.validation.get('cross_val_score'):
						score = Splits().cross_validation(kernelridge,self.trainF,self.trainR,self.validation,mission)
					elif self.validation.get('time_series_score'):
						score = Splits().time_series_validation(kernelridge,self.trainF,self.trainR,self.validation)
					elif self.validation.get('group_time_series_score'):
						trainTimeID = self.trainTimeID.apply(lambda x: int(str(x)[0:6]))
						score = Splits().group_time_series_validation(kernelridge,self.trainF,self.trainR,trainTimeID,self.validation)
					return abs(score)

				# default setting
				if len(kwargs) == 0:
					kernel_lst = ['linear','rbf','sigmoid']
					space = {'alpha': hp.uniform('alpha',1e-3,1e0),
							'gamma': hp.uniform('gamma',1e-2,1e+2),
							'kernel': hp.choice('kernel',kernel_lst)}
					optparams = fmin(fn=kernelridge,space=space,algo=tpe.suggest,max_evals=10)
					optparams.update({'kernel':kernel_lst[optparams['kernel']]})
				else:
					# get to which category the hyperparameter belongs
					uniform_lst,int_lst,choice_lst = at(self.model_params['kernelridge'],'uniform','int','choice')
					space,int_key,choice_key = {},[],[]
					for key,value in kwargs.items():
						# belonging to choice type
						if key in choice_lst or len(value) == 1:
							space.update({key:hp.choice(key,value)})
							choice_key.append((key,value))
						# belonging to float type
						elif key in uniform_lst:
							space.update({key:hp.uniform(key,value[0],value[1])})
						# belonging to float type
						elif key in int_lst:
							space.update({key:scope.int(hp.uniform(key,value[0],value[1]))})
							int_key.append(key)
						else:
							print('key %s does not belongs to uniform/int/choice, thus using default value instead'%(key))
					# get the dictionary of the optimal hyperparameters
					optparams = fmin(fn=kernelridge,space=space,algo=tpe.suggest,max_evals=10)
					for key in int_key:
						optparams[key] = int(optparams[key])
					for item in choice_key:
						optparams.update({item[0]:item[1][optparams[item[0]]]})
				# apply the optimal hyperparameters to the model and train it
				kernelRidge = KernelRidge()
				kernelRidge.set_params(**optparams)
				kernelRidgeTrained = kernelRidge.fit(self.trainF,self.trainR)
		else:
			raise ValueError('The mission of kernel ridge regression should be regression instead of %s'%(mission))

		return kernelRidgeTrained


	# ridge regression
	def ridge(self,mission='regression',**kwargs):
		if mission == 'regression':
			if self.paraTuneMethod == 'NoHyperTune':
				print('No hyperparameter optimization needed, now training the model')
				ridge = Ridge(**kwargs)
				ridgeTrained = ridge.fit(self.trainF,self.trainR)

			elif self.paraTuneMethod == 'gridsearchcv':
				print('use grid search method to do hyperparameter optimizaton, now training the model')
				ridge = Ridge()	

				if len(kwargs) == 0:
					paramSpace = {'alpha': [1e0, 0.1, 1e-2, 1e-3]
								}
				else:
					paramSpace = kwargs
				ridge = GridSearchCV(ridge,paramSpace,cv=4,n_jobs=-1)
				ridgeTrained = ridge.fit(self.trainF,self.trainR)

			elif self.paraTuneMethod == 'hyperopt':
				print('use bayesian hyperparameter optimization, now training the model')

				def ridge(paramSpace):
					ridge = Ridge()
					ridge.set_params(**paramSpace)
					if self.validation == None:
						score = cross_val_score(ridge,self.trainF,self.trainR,cv=4,scoring='neg_mean_squared_error').mean()
						print('use cross validation with parameters cv=4, scoring=neg_mean_squared_error')
					elif self.validation.get('cross_val_score'):
						score = Splits().cross_validation(ridge,self.trainF,self.trainR,self.validation,mission)
					elif self.validation.get('time_series_score'):
						score = Splits().time_series_validation(ridge,self.trainF,self.trainR,self.validation)
					elif self.validation.get('group_time_series_score'):
						trainTimeID = self.trainTimeID.apply(lambda x: int(str(x)[0:6]))
						score = Splits().group_time_series_validation(ridge,self.trainF,self.trainR,trainTimeID,self.validation)
					return abs(score)

				if len(kwargs) == 0:
					space = {'alpha': hp.uniform('alpha',1e-3,1e0)}
					optparams = fmin(fn=ridge,space=space,algo=tpe.suggest,max_evals=10)
				else:
					uniform_lst,int_lst,choice_lst = at(self.model_params['ridge']['regression'],'uniform','int','choice')
					space,int_key,choice_key = {},[],[]
					for key,value in kwargs.items():
						if key in choice_lst or len(value) == 1:
							space.update({key:hp.choice(key,value)})
							choice_key.append((key,value))
						elif key in uniform_lst:
							space.update({key:hp.uniform(key,value[0],value[1])})
						elif key in int_lst:
							space.update({key:scope.int(hp.uniform(key,value[0],value[1]))})
							int_key.append(key)
						else:
							print('key %s does not belongs to uniform/int/choice, thus using default value instead'%(key))
					optparams = fmin(fn=ridge,space=space,algo=tpe.suggest,max_evals=10)
					for key in int_key:
						optparams[key] = int(optparams[key])
					for item in choice_key:
						optparams.update({item[0]:item[1][optparams[item[0]]]})
				ridge = Ridge()
				ridge.set_params(**optparams)
				ridgeTrained = ridge.fit(self.trainF,self.trainR)

		elif mission == 'classification':
			if self.paraTuneMethod == 'NoHyperTune':
				print('No hyperparameter optimization needed, now training the model')
				ridge = RidgeClassifier(**kwargs)
				ridgeTrained = ridge.fit(self.trainF,self.trainL)

			elif self.paraTuneMethod == 'gridsearchcv':
				print('use grid search method to do hyperparameter optimizaton, now training the model')
				ridge = RidgeClassifier()

				if len(kwargs) == 0:
					paramSpace = {'alpha': [1e0, 0.1, 1e-2, 1e-3]
								}
				else:
					paramSpace = kwargs
				ridge = GridSearchCV(ridge,paramSpace,cv=4,n_jobs=-1)
				ridgeTrained = ridge.fit(self.trainF,self.trainL)

			elif self.paraTuneMethod == 'hyperopt':
				print('use bayesian hyperparameter optimization, now training the model')

				def ridge(paramSpace):
					ridge = RidgeClassifier()
					ridge.set_params(**paramSpace)
					if self.validation == None or (not self.validation.get('ridge')):
						score = cross_val_score(ridge,self.trainF,self.trainL,cv=4,scoring='roc_auc').mean()
						print('use cross validation with parameters cv=4, scoring=roc_auc')
					elif self.validation.get('cross_val_score'):
						score = Splits().cross_validation(ridge,self.trainF,self.trainL,self.validation,mission)
					elif self.validation.get('time_series_score'):
						score = Splits().time_series_validation(ridge,self.trainF,self.trainL,self.validation)
					elif self.validation.get('group_time_series_score'):
						trainTimeID = self.trainTimeID.apply(lambda x: int(str(x)[0:6]))
						score = Splits().group_time_series_validation(ridge,self.trainF,self.trainL,trainTimeID,self.validation)
					return -score

				if len(kwargs) == 0:
					space = {'alpha': hp.uniform('alpha',1e-3,1e0)}
					optparams = fmin(fn=ridge,space=space,algo=tpe.suggest,max_evals=10)
				else:
					uniform_lst,int_lst,choice_lst = at(self.model_params['ridge']['classification'],'uniform','int','choice')
					space,int_key,choice_key = {},[],[]
					for key,value in kwargs.items():
						if key in choice_lst or len(value) == 1:
							space.update({key:hp.choice(key,value)})
							choice_key.append((key,value))
						elif key in uniform_lst:
							space.update({key:hp.uniform(key,value[0],value[1])})
						elif key in int_lst:
							space.update({key:scope.int(hp.uniform(key,value[0],value[1]))})
							int_key.append(key)
						else:
							print('key %s does not belongs to uniform/int/choice, thus using default value instead'%(key))
					optparams = fmin(fn=ridge,space=space,algo=tpe.suggest,max_evals=10)
					for key in int_key:
						optparams[key] = int(optparams[key])
					for item in choice_key:
						optparams.update({item[0]:item[1][optparams[item[0]]]})
				ridge = RidgeClassifier()
				ridge.set_params(**optparams)
				ridgeTrained = ridge.fit(self.trainF,self.trainL)

		return ridgeTrained


	# lasso regression
	def lasso(self,mission='regression',**kwargs):
		if mission == 'regression':
			if self.paraTuneMethod == 'NoHyperTune':
				print('No hyperparameter optimization needed, now training the model')
				lasso = Lasso(**kwargs)
				lassoTrained = lasso.fit(self.trainF,self.trainR)

			elif self.paraTuneMethod == 'gridsearchcv':
				print('use grid search method to do hyperparameter optimizaton, now training the model')
				lasso = Lasso()	

				if len(kwargs) == 0:
					paramSpace = {'alpha': [1e0, 0.1, 1e-2, 1e-3]
								}
				else:
					paramSpace = kwargs
				lasso = GridSearchCV(lasso,paramSpace,cv=4,n_jobs=-1)
				lassoTrained = lasso.fit(self.trainF,self.trainR)

			elif self.paraTuneMethod == 'hyperopt':
				print('use bayesian hyperparameter optimization, now training the model')

				def lasso(paramSpace):
					lasso = Lasso()
					lasso.set_params(**paramSpace)
					if self.validation == None:
						score = cross_val_score(lasso,self.trainF,self.trainR,cv=4,scoring='neg_mean_squared_error').mean()
						print('use cross validation with parameters cv=4, scoring=neg_mean_squared_error')
					elif self.validation.get('cross_val_score'):
						score = Splits().cross_validation(lasso,self.trainF,self.trainR,self.validation,mission)
					elif self.validation.get('time_series_score'):
						score = Splits().time_series_validation(lasso,self.trainF,self.trainR,self.validation)
					elif self.validation.get('group_time_series_score'):
						trainTimeID = self.trainTimeID.apply(lambda x: int(str(x)[0:6]))
						score = Splits().group_time_series_validation(lasso,self.trainF,self.trainR,trainTimeID,self.validation)
					return abs(score)

				if len(kwargs) == 0:
					space = {'alpha': hp.uniform('alpha',1e-3,1e0)}
					optparams = fmin(fn=lasso,space=space,algo=tpe.suggest,max_evals=10)
				else:
					uniform_lst,int_lst,choice_lst = at(self.model_params['lasso'],'uniform','int','choice')
					space,int_key,choice_key = {},[],[]
					for key,value in kwargs.items():
						if key in choice_lst or len(value) == 1:
							space.update({key:hp.choice(key,value)})
							choice_key.append((key,value))
						elif key in uniform_lst:
							space.update({key:hp.uniform(key,value[0],value[1])})
						elif key in int_lst:
							space.update({key:scope.int(hp.uniform(key,value[0],value[1]))})
							int_key.append(key)
						else:
							print('key %s does not belongs to uniform/int/choice, thus using default value instead'%(key))
					optparams = fmin(fn=lasso,space=space,algo=tpe.suggest,max_evals=10)
					for key in int_key:
						optparams[key] = int(optparams[key])
					for item in choice_key:
						optparams.update({item[0]:item[1][optparams[item[0]]]})
				lasso = Lasso()
				lasso.set_params(**optparams)
				lassoTrained = lasso.fit(self.trainF,self.trainR)
		else:
			raise ValueError('The mission of lasso regression should be regression instead of %s'%(mission))

		return lassoTrained

	# elastic net
	def elasticNet(self,mission='regression',**kwargs):
		if mission == 'regression':
			if self.paraTuneMethod == 'NoHyperTune':
				print('No hyperparameter optimization needed, now training the model')
				elasticNet = ElasticNet(**kwargs)
				elasticNetTrained = elasticNet.fit(self.trainF,self.trainR)

			elif self.paraTuneMethod == 'gridsearchcv':
				print('use grid search method to do hyperparameter optimizaton, now training the model')
				elasticNet = ElasticNet()	

				if len(kwargs) == 0:
					paramSpace = {'alpha': [1e0, 0.1, 1e-2, 1e-3]
								}
				else:
					paramSpace = kwargs
				elasticNet = GridSearchCV(elasticNet,paramSpace,cv=4,n_jobs=-1)
				elasticNetTrained = elasticNet.fit(self.trainF,self.trainR)

			elif self.paraTuneMethod == 'hyperopt':
				print('use bayesian hyperparameter optimization, now training the model')

				def elasticNet(paramSpace):
					elasticNet = ElasticNet()
					elasticNet.set_params(**paramSpace)
					if self.validation == None:
						score = cross_val_score(elasticNet,self.trainF,self.trainR,cv=4,scoring='neg_mean_squared_error').mean()
						print('use cross validation with parameters cv=4, scoring=neg_mean_squared_error')
					elif self.validation.get('cross_val_score'):
						score = Splits().cross_validation(elasticNet,self.trainF,self.trainR,self.validation,mission)
					elif self.validation.get('time_series_score'):
						score = Splits().time_series_validation(elasticNet,self.trainF,self.trainR,self.validation)
					elif self.validation.get('group_time_series_score'):
						trainTimeID = self.trainTimeID.apply(lambda x: int(str(x)[0:6]))
						score = Splits().group_time_series_validation(elasticNet,self.trainF,self.trainR,trainTimeID,self.validation)
					return abs(score)

				if len(kwargs) == 0:
					space = {'alpha': hp.uniform('alpha',1e-3,1e0)}
					optparams = fmin(fn=elasticNet,space=space,algo=tpe.suggest,max_evals=10)
				else:
					uniform_lst,int_lst,choice_lst = at(self.model_params['elasticnet'],'uniform','int','choice')
					space,int_key,choice_key = {},[],[]
					for key,value in kwargs.items():
						if key in choice_lst or len(value) == 1:
							space.update({key:hp.choice(key,value)})
							choice_key.append((key,value))
						elif key in uniform_lst:
							space.update({key:hp.uniform(key,value[0],value[1])})
						elif key in int_lst:
							space.update({key:scope.int(hp.uniform(key,value[0],value[1]))})
							int_key.append(key)
						else:
							print('key %s does not belongs to uniform/int/choice, thus using default value instead'%(key))
					optparams = fmin(fn=elasticNet,space=space,algo=tpe.suggest,max_evals=10)
					for key in int_key:
						optparams[key] = int(optparams[key])
					for item in choice_key:
						optparams.update({item[0]:item[1][optparams[item[0]]]})
				elasticNet = ElasticNet()
				elasticNet.set_params(**optparams)
				elasticNetTrained = elasticNet.fit(self.trainF,self.trainR)
		else:
			raise ValueError('The mission of elastic net should be regression instead of %s'%(mission))

		return elasticNetTrained

	# support vector regression
	def svr(self,mission='regression',**kwargs):
		if mission == 'regression':
			if self.paraTuneMethod == 'NoHyperTune':
				print('No hyperparameter optimization needed, now training the model')
				svr = SVR(**kwargs)
				svrTrained = svr.fit(self.trainF,self.trainR)

			elif self.paraTuneMethod == 'gridsearchcv':
				print('use grid search method to do hyperparameter optimizaton, now training the model')
				svr = SVR()	

				if len(kwargs) == 0:
					paramSpace = {'C': [1e0, 0.1, 1e-2, 1e-3], # the reciprocal of the coefficient of regularization term
								'gamma': np.logspace(-2, 2, 5),
								'kernel':'rbf'
								}
				else:
					paramSpace = kwargs
				svr = GridSearchCV(svr,paramSpace,cv=4,n_jobs=-1)
				svrTrained = svr.fit(self.trainF,self.trainR)

			elif self.paraTuneMethod == 'hyperopt':
				print('use bayesian hyperparameter optimization, now training the model')

				def svr(paramSpace):
					svr = SVR()
					svr.set_params(**paramSpace)
					if self.validation == None:
						score = cross_val_score(svr,self.trainF,self.trainR,cv=4,scoring='neg_mean_squared_error').mean()
						print('use cross validation with parameters cv=4, scoring=neg_mean_squared_error')
					elif self.validation.get('cross_val_score'):
						score = Splits().cross_validation(svr,self.trainF,self.trainR,self.validation,mission)
					elif self.validation.get('time_series_score'):
						score = Splits().time_series_validation(svr,self.trainF,self.trainR,self.validation)
					elif self.validation.get('group_time_series_score'):
						trainTimeID = self.trainTimeID.apply(lambda x: int(str(x)[0:6]))
						score = Splits().group_time_series_validation(svr,self.trainF,self.trainR,trainTimeID,self.validation)
					return abs(score)

				kernel_lst = ['linear','rbf','sigmoid']
				space = {'C': hp.uniform('C',1e-3,1e0),
						'gamma': hp.uniform('gamma',1e-2,1e+2),
						'kernel': hp.choice('kernel',kernel_lst)}
				if len(kwargs) == 0:
					optparams = fmin(fn=svr,space=space,algo=tpe.suggest,max_evals=10)
					optparams.update({'kernel':kernel_lst[optparams['kernel']]})
				else:
					uniform_lst,int_lst,choice_lst = at(self.model_params['svr'],'uniform','int','choice')
					space,int_key,choice_key = {},[],[]
					for key,value in kwargs.items():
						if key in choice_lst or len(value) == 1:
							space.update({key:hp.choice(key,value)})
							choice_key.append((key,value))
						elif key in uniform_lst:
							space.update({key:hp.uniform(key,value[0],value[1])})
						elif key in int_lst:
							space.update({key:scope.int(hp.uniform(key,value[0],value[1]))})
							int_key.append(key)
						else:
							print('key %s does not belongs to uniform/int/choice, thus using default value instead'%(key))
					optparams = fmin(fn=svr,space=space,algo=tpe.suggest,max_evals=10)
					for key in int_key:
						optparams[key] = int(optparams[key])
					for item in choice_key:
						optparams.update({item[0]:item[1][optparams[item[0]]]})
				svr = SVR()
				svr.set_params(**optparams)
				svrTrained = svr.fit(self.trainF,self.trainR)
		else:
			raise ValueError('The mission of support vector regression should be regression instead of %s'%(mission))

		return svrTrained

	# random forest
	def randomForest(self,mission='classification',**kwargs):
		if mission == 'classification':
			if self.paraTuneMethod == 'NoHyperTune':
				print('No hyperparameter optimization needed, now training the model')
				randomForest = RandomForestClassifier(**kwargs)
				randomForestTrained = randomForest.fit(self.trainF,self.trainL)

			elif self.paraTuneMethod == 'gridsearchcv':
				print('use grid search method to do hyperparameter optimizaton, now training the model')
				randomForest = RandomForestClassifier()	

				if len(kwargs) == 0:
					paramSpace = {'n_estimators': [200,300,400], # the number of tree
								'max_depth': [None], # the maximum depth of tree
								'min_samples_split':[2] # the minimum number of sample required for splitting a node
								}
				else:
					paramSpace = kwargs
				randomForest = GridSearchCV(randomForest,paramSpace,cv=4,n_jobs=-1)
				randomForestTrained = randomForest.fit(self.trainF,self.trainL)

			elif self.paraTuneMethod == 'hyperopt':
				print('use bayesian hyperparameter optimization, now training the model')
				
				def randomforest(paramSpace):
					randomForest = RandomForestClassifier()
					randomForest.set_params(**paramSpace)
					if self.validation == None:
						score = cross_val_score(randomForest,self.trainF,self.trainL,cv=4,scoring='roc_auc').mean()
						print('use cross validation with parameters cv=4, scoring=roc_auc')
					elif self.validation.get('cross_val_score'):
						score = Splits().cross_validation(randomforest,self.trainF,self.trainL,self.validation,mission)
					elif self.validation.get('time_series_score'):
						score = Splits().time_series_validation(randomforest,self.trainF,self.trainL,self.validation)
					elif self.validation.get('group_time_series_score'):
						trainTimeID = self.trainTimeID.apply(lambda x: int(str(x)[0:6]))
						score = Splits().group_time_series_validation(randomforest,self.trainF,self.trainL,trainTimeID,self.validation)
					return -score
					
				if len(kwargs) == 0:
					space = {'n_estimators': scope.int(hp.uniform('n_estimators',200,400)),
							'max_depth': scope.int(hp.uniform('max_depth',2,5)),
							'min_samples_split': scope.int(hp.uniform('min_samples_split',2,5))}
					optparams = fmin(fn=randomforest,space=space,algo=tpe.suggest,max_evals=10)
				else:
					uniform_lst,int_lst,choice_lst = at(self.model_params['randomforest']['classification'],'uniform','int','choice')
					space,int_key,choice_key = {},[],[]
					for key,value in kwargs.items():
						if key in choice_lst or len(value) == 1:
							space.update({key:hp.choice(key,value)})
							choice_key.append((key,value))
						elif key in uniform_lst:
							space.update({key:hp.uniform(key,value[0],value[1])})
						elif key in int_lst:
							space.update({key:scope.int(hp.uniform(key,value[0],value[1]))})
							int_key.append(key)
						else:
							print('key %s does not belongs to uniform/int/choice, thus using default value instead'%(key))
					optparams = fmin(fn=randomforest,space=space,algo=tpe.suggest,max_evals=10)
					for key in int_key:
						optparams[key] = int(optparams[key])
					for item in choice_key:
						optparams.update({item[0]:item[1][optparams[item[0]]]})
				randomForest = RandomForestClassifier()
				randomForest.set_params(**optparams)
				randomForestTrained = randomForest.fit(self.trainF,self.trainL)

		elif mission == 'regression':
			if self.paraTuneMethod == 'NoHyperTune':
				print('No hyperparameter optimization needed, now training the model')
				randomForest = RandomForestRegressor(**kwargs)
				randomForestTrained = randomForest.fit(self.trainF,self.trainR)

			elif self.paraTuneMethod == 'gridsearchcv':
				print('use grid search method to do hyperparameter optimizaton, now training the model')
				randomForest = RandomForestRegressor()

				if len(kwargs) == 0:
					paramSpace = {'n_estimators': [200,300,400],
								'max_depth': [None], 
								'min_samples_split':[2] 
								}
				else:
					paramSpace = kwargs
				randomForest = GridSearchCV(randomForest,paramSpace,cv=4,n_jobs=-1)
				randomForestTrained = randomForest.fit(self.trainF,self.trainR)

			elif self.paraTuneMethod == 'hyperopt':
				print('use bayesian hyperparameter optimization, now training the model')
				
				def randomforest(paramSpace):
					randomforest = RandomForestRegressor()
					randomforest.set_params(**paramSpace)
					if self.validation == None:
						score = cross_val_score(randomforest,self.trainF,self.trainR,cv=4,scoring='neg_mean_squared_error').mean()
						print('use cross validation with parameters cv=4, scoring=neg_mean_squared_error')
					elif self.validation.get('cross_val_score'):
						score = Splits().cross_validation(randomforest,self.trainF,self.trainR,self.validation,mission)
					elif self.validation.get('time_series_score'):
						score = Splits().time_series_validation(randomforest,self.trainF,self.trainR,self.validation)
					elif self.validation.get('group_time_series_score'):
						trainTimeID = self.trainTimeID.apply(lambda x: int(str(x)[0:6]))
						score = Splits().group_time_series_validation(randomforest,self.trainF,self.trainR,trainTimeID,self.validation)
					return abs(score)
					
				if len(kwargs) == 0:
					space = {'n_estimators': scope.int(hp.uniform('n_estimators',200,400)),
							'max_depth': scope.int(hp.uniform('max_depth',2,5)),
							'min_samples_split': scope.int(hp.uniform('min_samples_split',2,5))}
					optparams = fmin(fn=randomforest,space=space,algo=tpe.suggest,max_evals=10)
				else:
					uniform_lst,int_lst,choice_lst = at(self.model_params['randomforest']['regression'],'uniform','int','choice')
					space,int_key,choice_key = {},[],[]
					for key,value in kwargs.items():
						if key in choice_lst or len(value) == 1:
							space.update({key:hp.choice(key,value)})
							choice_key.append((key,value))
						elif key in uniform_lst:
							space.update({key:hp.uniform(key,value[0],value[1])})
						elif key in int_lst:
							space.update({key:scope.int(hp.uniform(key,value[0],value[1]))})
							int_key.append(key)
						else:
							print('key %s does not belongs to uniform/int/choice, thus using default value instead'%(key))
					optparams = fmin(fn=randomforest,space=space,algo=tpe.suggest,max_evals=10)
					for key in int_key:
						optparams[key] = int(optparams[key])
					for item in choice_key:
						optparams.update({item[0]:item[1][optparams[item[0]]]})
				randomForest = RandomForestRegressor()
				randomForest.set_params(**optparams)
				randomForestTrained = randomForest.fit(self.trainF,self.trainR)
		else:
			raise ValueError('The mission of random forest should be regression or classification instead of %s'%(mission))

		return randomForestTrained

	# gradient boosting decision tree
	def gbdt(self,mission='classification',**kwargs):
		if mission == 'classification':
			if self.paraTuneMethod == 'NoHyperTune':
				print('No hyperparameter optimization needed, now training the model')
				gbdt = GradientBoostingClassifier(**kwargs)
				gbdtTrained = gbdt.fit(self.trainF,self.trainL)
				
			elif self.paraTuneMethod == 'gridsearchcv':
				print('use grid search method to do hyperparameter optimizaton, now training the model')
				gbdt = GradientBoostingClassifier()	

				if len(kwargs) == 0:
					paramSpace = {'n_estimators': [200,300,400], # the number of estimator
								'max_depth': [2,3,4,5], 
								'learning_rate':[0.1,0.5,1.0] # learning rate, served as the contriution of tree
								}
				else:
					paramSpace = kwargs
				gbdt = GridSearchCV(gbdt,paramSpace,cv=4,n_jobs=-1)
				gbdtTrained = gbdt.fit(self.trainF,self.trainL)

			elif self.paraTuneMethod == 'hyperopt':
				print('use bayesian hyperparameter optimization, now training the model')
				
				def gbdt(paramSpace):
					gbdt = GradientBoostingClassifier()
					gbdt.set_params(**paramSpace)
					if self.validation == None:
						score = cross_val_score(gbdt,self.trainF,self.trainL,cv=4,scoring='roc_auc').mean()
						print('use cross validation with parameters cv=4, scoring=roc_auc')
					elif self.validation.get('cross_val_score'):
						score = Splits().cross_validation(gbdt,self.trainF,self.trainL,self.validation,mission)
					elif self.validation.get('time_series_score'):
						score = Splits().time_series_validation(gbdt,self.trainF,self.trainL,self.validation)
					elif self.validation.get('group_time_series_score'):
						trainTimeID = self.trainTimeID.apply(lambda x: int(str(x)[0:6]))
						score = Splits().group_time_series_validation(gbdt,self.trainF,self.trainL,trainTimeID,self.validation)
					return -score

				if len(kwargs) == 0:
					space = {'n_estimators': scope.int(hp.uniform('n_estimators',200,400)),
							'max_depth': scope.int(hp.uniform('max_depth',2,5)),
							'learning_rate': hp.uniform('learning_rate',0.1,1.0)}
					optparams = fmin(fn=gbdt,space=space,algo=tpe.suggest,max_evals=10)
				else:
					uniform_lst,int_lst,choice_lst = at(self.model_params['gbdt']['classification'],'uniform','int','choice')
					space,int_key,choice_key = {},[],[]
					for key,value in kwargs.items():
						if key in choice_lst or len(value) == 1:
							space.update({key:hp.choice(key,value)})
							choice_key.append((key,value))
						elif key in uniform_lst:
							space.update({key:hp.uniform(key,value[0],value[1])})
						elif key in int_lst:
							space.update({key:scope.int(hp.uniform(key,value[0],value[1]))})
							int_key.append(key)
						else:
							print('key %s does not belongs to uniform/int/choice, thus using default value instead'%(key))
					optparams = fmin(fn=gbdt,space=space,algo=tpe.suggest,max_evals=10)
					for key in int_key:
						optparams[key] = int(optparams[key])
					for item in choice_key:
						optparams.update({item[0]:item[1][optparams[item[0]]]})
				gbdt = GradientBoostingClassifier()
				gbdt.set_params(**optparams)
				gbdtTrained = gbdt.fit(self.trainF,self.trainL)

		elif mission == 'regression':
			if self.paraTuneMethod == 'NoHyperTune':
				print('No hyperparameter optimization needed, now training the model')
				gbdt = GradientBoostingRegressor(**kwargs)
				gbdtTrained = gbdt.fit(self.trainF,self.trainR)
				
			elif self.paraTuneMethod == 'gridsearchcv':
				print('use grid search method to do hyperparameter optimizaton, now training the model')
				gbdt = GradientBoostingRegressor()

				if len(kwargs) == 0:
					paramSpace = {
						'n_estimators': [200,300,400],
						'max_depth': [2,3,4,5], 
						'learning_rate':[0.1,0.5,1.0] 
								}
				else:
					paramSpace = kwargs
				gbdt = GridSearchCV(gbdt,paramSpace,cv=4,n_jobs=-1)
				gbdtTrained = gbdt.fit(self.trainF,self.trainR)

			elif self.paraTuneMethod == 'hyperopt':
				print('use bayesian hyperparameter optimization, now training the model')
				
				def gbdt(paramSpace):
					gbdt = GradientBoostingRegressor()
					gbdt.set_params(**paramSpace)
					if self.validation == None:
						score = cross_val_score(gbdt,self.trainF,self.trainR,cv=4,scoring='neg_mean_squared_error').mean()
						print('use cross validation with parameters cv=4, scoring=neg_mean_squared_error')
					elif self.validation.get('cross_val_score'):
						score = Splits().cross_validation(gbdt,self.trainF,self.trainR,self.validation,mission)
					elif self.validation.get('time_series_score'):
						score = Splits().time_series_validation(gbdt,self.trainF,self.trainR,self.validation)
					elif self.validation.get('group_time_series_score'):
						trainTimeID = self.trainTimeID.apply(lambda x: int(str(x)[0:6]))
						score = Splits().group_time_series_validation(gbdt,self.trainF,self.trainR,trainTimeID,self.validation)
					return abs(score)
					
				if len(kwargs) == 0:
					space = {'n_estimators': scope.int(hp.uniform('n_estimators',200,400)),
							'max_depth': scope.int(hp.uniform('max_depth',2,5)),
							'learning_rate': hp.uniform('learning_rate',0.1,1.0)}
					optparams = fmin(fn=gbdt,space=space,algo=tpe.suggest,max_evals=10)
				else:
					uniform_lst,int_lst,choice_lst = at(self.model_params['gbdt']['regression'],'uniform','int','choice')
					space,int_key,choice_key = {},[],[]
					for key,value in kwargs.items():
						if key in choice_lst or len(value) == 1:
							space.update({key:hp.choice(key,value)})
							choice_key.append((key,value))
						elif key in uniform_lst:
							space.update({key:hp.uniform(key,value[0],value[1])})
						elif key in int_lst:
							space.update({key:scope.int(hp.uniform(key,value[0],value[1]))})
							int_key.append(key)
						else:
							print('key %s does not belongs to uniform/int/choice, thus using default value instead'%(key))
					optparams = fmin(fn=gbdt,space=space,algo=tpe.suggest,max_evals=10)
					for key in int_key:
						optparams[key] = int(optparams[key])
					for item in choice_key:
						optparams.update({item[0]:item[1][optparams[item[0]]]})
				gbdt = GradientBoostingRegressor()
				for key in ['n_estimators', 'max_depth']:
					optparams[key] = int(optparams[key])
				gbdt.set_params(**optparams)
				gbdtTrained = gbdt.fit(self.trainF,self.trainR)

		else:
			raise ValueError('The mission of gradient boosting decision tree should be regression or classification instead of %s'%(mission))

		return gbdtTrained

	# multilayer perceptron
	def mlpc(self,mission='classification',**kwargs):
		if mission == 'classification':
			if self.paraTuneMethod == 'NoHyperTune':
				print('No hyperparameter optimization needed, now training the model')
				mlpc = MLPClassifier(**kwargs)
				mlpcTrained = mlpc.fit(self.trainF,self.trainL)

			elif self.paraTuneMethod == 'gridsearchcv':
				print('use grid search method to do hyperparameter optimizaton, now training the model')
				mlpc = MLPClassifier()	
				if len(kwargs) == 0:
					paramSpace = {'alpha': [1e0, 0.1, 1e-2, 1e-3]} # the coefficient of l2 regularization
				else:
					paramSpace = kwargs
				mlpc = GridSearchCV(mlpc,paramSpace,cv=4,n_jobs=-1)
				mlpcTrained = mlpc.fit(self.trainF,self.trainL)

			elif self.paraTuneMethod == 'hyperopt':
				print('use bayesian hyperparameter optimization, now training the model')
				
				def mlpc(paramSpace):
					mlpc = MLPClassifier()
					mlpc.set_params(**paramSpace)
					if self.validation == None:
						score = cross_val_score(mlpc,self.trainF,self.trainL,cv=4,scoring='roc_auc').mean()
						print('use cross validation with parameters cv=4, scoring=roc_auc')
					elif self.validation.get('cross_val_score'):
						score = Splits().cross_validation(mlpc,self.trainF,self.trainL,self.validation,mission)
					elif self.validation.get('time_series_score'):
						score = Splits().time_series_validation(mlpc,self.trainF,self.trainL,self.validation)
					elif self.validation.get('group_time_series_score'):
						trainTimeID = self.trainTimeID.apply(lambda x: int(str(x)[0:6]))
						score = Splits().group_time_series_validation(mlpc,self.trainF,self.trainL,trainTimeID,self.validation)
					return -score
					
				if len(kwargs) == 0:
					space = {'alpha':hp.uniform('alpha',1e-3,1e0)}
					optparams = fmin(fn=mlpc,space=space,algo=tpe.suggest,max_evals=10)
				else:
					uniform_lst,int_lst,choice_lst = at(self.model_params['mlpc']['classification'],'uniform','int','choice')
					space,int_key,choice_key = {},[],[]
					for key,value in kwargs.items():
						if key in choice_lst or len(value) == 1:
							space.update({key:hp.choice(key,value)})
							choice_key.append((key,value))
						elif key in uniform_lst:
							space.update({key:hp.uniform(key,value[0],value[1])})
						elif key in int_lst:
							space.update({key:scope.int(hp.uniform(key,value[0],value[1]))})
							int_key.append(key)
						else:
							print('key %s does not belongs to uniform/int/choice, thus using default value instead'%(key))
					optparams = fmin(fn=mlpc,space=space,algo=tpe.suggest,max_evals=10)
					for key in int_key:
						optparams[key] = int(optparams[key])
					for item in choice_key:
						optparams.update({item[0]:item[1][optparams[item[0]]]})
				mlpc = MLPClassifier()
				mlpc.set_params(**optparams)
				mlpcTrained = mlpc.fit(self.trainF,self.trainL)

		elif mission == 'regression':
			if self.paraTuneMethod == 'NoHyperTune':
				print('No hyperparameter optimization needed, now training the model')
				mlpc = MLPRegressor(**kwargs)
				mlpcTrained = mlpc.fit(self.trainF,self.trainR)

			elif self.paraTuneMethod == 'gridsearchcv':
				print('use grid search method to do hyperparameter optimizaton, now training the model')
				mlpc = MLPRegressor()	

				if len(kwargs) == 0:
					paramSpace = {'alpha': [1e0, 0.1, 1e-2, 1e-3]}
				else:
					paramSpace = kwargs
				mlpc = GridSearchCV(mlpc,paramSpace,cv=4,n_jobs=-1)
				mlpcTrained = mlpc.fit(self.trainF,self.trainR)

			elif self.paraTuneMethod == 'hyperopt':
				print('use bayesian hyperparameter optimization, now training the model')
				
				def mlpc(paramSpace):
					mlpc = MLPRegressor()
					mlpc.set_params(**paramSpace)
					if self.validation == None:
						score = cross_val_score(mlpc,self.trainF,self.trainR,cv=4,scoring='neg_mean_squared_error').mean()
						print('use cross validation with parameters cv=4, scoring=neg_mean_squared_error')
					elif self.validation.get('time_series_score'):
						score = Splits().time_series_validation(mlpc,self.trainF,self.trainR,self.validation)
					elif self.validation.get('group_time_series_score'):
						trainTimeID = self.trainTimeID.apply(lambda x: int(str(x)[0:6]))
						score = Splits().group_time_series_validation(mlpc,self.trainF,self.trainR,trainTimeID,self.validation)
					return abs(score)
					
				if len(kwargs) == 0:
					space = {'alpha':hp.uniform('alpha',1e-3,1e0)}
					optparams = fmin(fn=mlpc,space=space,algo=tpe.suggest,max_evals=10)
				else:
					uniform_lst,int_lst,choice_lst = at(self.model_params['mlpc']['regression'],'uniform','int','choice')
					space,int_key,choice_key = {},[],[]
					for key,value in kwargs.items():
						if key in choice_lst or len(value) == 1:
							space.update({key:hp.choice(key,value)})
							choice_key.append((key,value))
						elif key in uniform_lst:
							space.update({key:hp.uniform(key,value[0],value[1])})
						elif key in int_lst:
							space.update({key:scope.int(hp.uniform(key,value[0],value[1]))})
							int_key.append(key)
						else:
							print('key %s does not belongs to uniform/int/choice, thus using default value instead'%(key))
					optparams = fmin(fn=mlpc,space=space,algo=tpe.suggest,max_evals=10)
					for key in int_key:
						optparams[key] = int(optparams[key])
					for item in choice_key:
						optparams.update({item[0]:item[1][optparams[item[0]]]})
				mlpc = MLPRegressor()
				mlpc.set_params(**optparams)
				mlpcTrained = mlpc.fit(self.trainF,self.trainR)
		return mlpcTrained
		
	# gaussian naive bayes
	def gnb(self,mission='classification',**kwargs):
		if mission == 'classification':
			if self.paraTuneMethod == 'NoHyperTune':
				print('No hyperparameter optimization needed, now training the model')
				gnb = GaussianNB(**kwargs)
				gnbTrained = gnb.fit(self.trainF,self.trainL)

			elif self.paraTuneMethod == 'gridsearchcv':
				print('use grid search method to do hyperparameter optimizaton, now training the model')
				gnb = GaussianNB()	

				if len(kwargs) == 0:
					paramSpace = {'var_smoothing': [1e-09, 1e-07, 1e-05, 1e-03]}
				else:
					paramSpace = kwargs
				gnb = GridSearchCV(gnb,paramSpace,cv=4,n_jobs=-1)
				gnbTrained = gnb.fit(self.trainF,self.trainL)

			elif self.paraTuneMethod == 'hyperopt':
				print('use bayesian hyperparameter optimization, now training the model')
				
				def gnb(paramSpace):
					gnb = GaussianNB()
					gnb.set_params(**paramSpace)
					if self.validation == None:
						score = cross_val_score(gnb,self.trainF,self.trainL,cv=4,scoring='roc_auc').mean()
						print('use cross validation with parameters cv=4, scoring=roc_auc')
					elif self.validation.get('cross_val_score'):
						score = Splits().cross_validation(gnb,self.trainF,self.trainL,self.validation,mission)
					elif self.validation.get('time_series_score'):
						score = Splits().time_series_validation(gnb,self.trainF,self.trainL,self.validation)
					elif self.validation.get('group_time_series_score'):
						trainTimeID = self.trainTimeID.apply(lambda x: int(str(x)[0:6]))
						score = Splits().group_time_series_validation(gnb,self.trainF,self.trainL,trainTimeID,self.validation)
					return -score
					
				if len(kwargs) == 0:
					space = {'var_smoothing':hp.uniform('var_smoothing',1e-09,1e-03)}
					optparams = fmin(fn=gnb,space=space,algo=tpe.suggest,max_evals=10)
				else:
					uniform_lst,int_lst,choice_lst = at(self.model_params['gnb'],'uniform','int','choice')
					space,int_key,choice_key = {},[],[]
					for key,value in kwargs.items():
						if key in choice_lst or len(value) == 1:
							space.update({key:hp.choice(key,value)})
							choice_key.append((key,value))
						elif key in uniform_lst:
							space.update({key:hp.uniform(key,value[0],value[1])})
						elif key in int_lst:
							space.update({key:scope.int(hp.uniform(key,value[0],value[1]))})
							int_key.append(key)
						else:
							print('key %s does not belongs to uniform/int/choice, thus using default value instead'%(key))
					optparams = fmin(fn=gnb,space=space,algo=tpe.suggest,max_evals=10)
					for key in int_key:
						optparams[key] = int(optparams[key])
					for item in choice_key:
						optparams.update({item[0]:item[1][optparams[item[0]]]})
				gnb = GaussianNB()
				gnb.set_params(**optparams)
				gnbTrained = gnb.fit(self.trainF,self.trainL)
		else:
			raise ValueError('The mission of gaussian naive bayes should be classification instead of %s'%(mission))

		return gnbTrained

	# adaboost
	def adaboost(self,mission='classification',**kwargs):
		if mission == 'classification':
			if self.paraTuneMethod == 'NoHyperTune':
				print('No hyperparameter optimization needed, now training the model')
				adaboost = AdaBoostClassifier(**kwargs)
				adaboostTrained = adaboost.fit(self.trainF,self.trainL)

			elif self.paraTuneMethod == 'gridsearchcv':
				print('use grid search method to do hyperparameter optimizaton, now training the model')
				adaboost = AdaBoostClassifier()	

				if len(kwargs) == 0:
					paramSpace = {'learning_rate':[0.1,0.5,1.0],
									'n_estimators':[100,200,300,400]} 
				else:
					paramSpace = kwargs
				adaboost = GridSearchCV(adaboost,paramSpace,cv=4,n_jobs=-1)
				adaboostTrained = adaboost.fit(self.trainF,self.trainL)

			elif self.paraTuneMethod == 'hyperopt':
				print('use bayesian hyperparameter optimization, now training the model')
				
				def adaboost(paramSpace):
					adaboost = AdaBoostClassifier()
					adaboost.set_params(**paramSpace)
					if self.validation == None:
						score = cross_val_score(adaboost,self.trainF,self.trainL,cv=4,scoring='roc_auc').mean()
						print('use cross validation with parameters cv=4, scoring=roc_auc')
					elif self.validation.get('cross_val_score'):
						score = Splits().cross_validation(adaboost,self.trainF,self.trainL,self.validation,mission)
					elif self.validation.get('time_series_score'):
						score = Splits().time_series_validation(adaboost,self.trainF,self.trainL,self.validation)
					elif self.validation.get('group_time_series_score'):
						trainTimeID = self.trainTimeID.apply(lambda x: int(str(x)[0:6]))
						score = Splits().group_time_series_validation(adaboost,self.trainF,self.trainL,trainTimeID,self.validation)
					return -score
					
				if len(kwargs) == 0:
					space = {'learning_rate': hp.uniform('learning_rate',0.1,1.0),
							'n_estimators': scope.int(hp.uniform('n_estimators',100,400))}
					optparams = fmin(fn=adaboost,space=space,algo=tpe.suggest,max_evals=10)
				else:
					uniform_lst,int_lst,choice_lst = at(self.model_params['adaboost']['classification'],'uniform','int','choice')
					space,int_key,choice_key = {},[],[]
					for key,value in kwargs.items():
						if key in choice_lst or len(value) == 1:
							space.update({key:hp.choice(key,value)})
							choice_key.append((key,value))
						elif key in uniform_lst:
							space.update({key:hp.uniform(key,value[0],value[1])})
						elif key in int_lst:
							space.update({key:scope.int(hp.uniform(key,value[0],value[1]))})
							int_key.append(key)
						else:
							print('key %s does not belongs to uniform/int/choice, thus using default value instead'%(key))
					optparams = fmin(fn=adaboost,space=space,algo=tpe.suggest,max_evals=10)
					for key in int_key:
						optparams[key] = int(optparams[key])
					for item in choice_key:
						optparams.update({item[0]:item[1][optparams[item[0]]]})
				adaboost = AdaBoostClassifier()
				adaboost.set_params(**optparams)
				adaboostTrained = adaboost.fit(self.trainF,self.trainL)

		elif mission == 'regression':
			if self.paraTuneMethod == 'NoHyperTune':
				print('No hyperparameter optimization needed, now training the model')
				adaboost = AdaBoostRegressor(**kwargs)
				adaboostTrained = adaboost.fit(self.trainF,self.trainR)

			elif self.paraTuneMethod == 'gridsearchcv':
				print('use grid search method to do hyperparameter optimizaton, now training the model')
				adaboost = AdaBoostRegressor()	

				if len(kwargs) == 0:
					paramSpace = {'learning_rate':[0.1,0.5,1.0], 
									'n_estimators':[100,200,300,400]} 
				else:
					paramSpace = kwargs
				adaboost = GridSearchCV(adaboost,paramSpace,cv=4,n_jobs=-1)
				adaboostTrained = adaboost.fit(self.trainF,self.trainR)

			elif self.paraTuneMethod == 'hyperopt':
				print('use bayesian hyperparameter optimization, now training the model')
				
				def adaboost(paramSpace):
					adaboost = AdaBoostRegressor()
					adaboost.set_params(**paramSpace)
					if self.validation == None:
						score = cross_val_score(adaboost,self.trainF,self.trainR,cv=4,scoring='neg_mean_squared_error').mean()
						print('use cross validation with parameters cv=4, scoring=neg_mean_squared_error')
					elif self.validation.get('cross_val_score'):
						score = Splits().cross_validation(adaboost,self.trainF,self.trainR,self.validation,mission)
					elif self.validation.get('time_series_score'):
						score = Splits().time_series_validation(adaboost,self.trainF,self.trainR,self.validation)
					elif self.validation.get('group_time_series_score'):
						trainTimeID = self.trainTimeID.apply(lambda x: int(str(x)[0:6]))
						score = Splits().group_time_series_validation(adaboost,self.trainF,self.trainR,trainTimeID,self.validation)
					return abs(score)

				if len(kwargs) == 0:
					space = {'learning_rate': hp.uniform('learning_rate',0.1,1.0),
							'n_estimators': scope.int(hp.uniform('n_estimators',100,400))}
					optparams = fmin(fn=adaboost,space=space,algo=tpe.suggest,max_evals=10)
				else:
					uniform_lst,int_lst,choice_lst = at(self.model_params['adaboost']['regression'],'uniform','int','choice')
					space,int_key,choice_key = {},[],[]
					for key,value in kwargs.items():
						if key in choice_lst or len(value) == 1:
							space.update({key:hp.choice(key,value)})
							choice_key.append((key,value))
						elif key in uniform_lst:
							space.update({key:hp.uniform(key,value[0],value[1])})
						elif key in int_lst:
							space.update({key:scope.int(hp.uniform(key,value[0],value[1]))})
							int_key.append(key)
						else:
							print('key %s does not belongs to uniform/int/choice, thus using default value instead'%(key))
					optparams = fmin(fn=adaboost,space=space,algo=tpe.suggest,max_evals=10)
					for key in int_key:
						optparams[key] = int(optparams[key])
					for item in choice_key:
						optparams.update({item[0]:item[1][optparams[item[0]]]})
				adaboost = AdaBoostRegressor()
				adaboost.set_params(**optparams)
				adaboostTrained = adaboost.fit(self.trainF,self.trainR)

		else:
			raise ValueError('The mission of adaboost should be regression or classification instead of %s'%(mission))

		return adaboostTrained

	# xgboost
	def xgboost(self,mission='classification',**kwargs):
		if mission == 'classification':
			if self.paraTuneMethod == 'NoHyperTune':
				print('No hyperparameter optimization needed, now training the model')
				xgboost = XGBClassifier(eval_metric='mlogloss',use_label_encoder=False,**kwargs)
				xgboostTrained = xgboost.fit(self.trainF,self.trainL)

			elif self.paraTuneMethod == 'gridsearchcv':
				print('use grid search method to do hyperparameter optimizaton, now training the model')
				xgboost = XGBClassifier(eval_metric='mlogloss',use_label_encoder=False)

				if len(kwargs) == 0:
					paramSpace = {'n_jobs':[-1],
									'learning_rate':[0.1,0.5,1.0], 
									'n_estimators':[200,300,400], 
									'max_depth':[3,4,5,6], 
									'objective': ['binary:logistic']} 
				else:
					paramSpace = kwargs
				xgboost = GridSearchCV(xgboost,paramSpace,cv=4,n_jobs=-1)
				xgboostTrained = xgboost.fit(self.trainF,self.trainL)

			elif self.paraTuneMethod == 'hyperopt':
				print('use bayesian hyperparameter optimization, now training the model')
				
				def xgboost(paramSpace):
					xgboost = XGBClassifier()
					xgboost.set_params(**paramSpace)
					if self.validation == None:
						score = cross_val_score(xgboost,self.trainF,self.trainL,cv=4,scoring='roc_auc').mean()
						print('use cross validation with parameters cv=4, scoring=roc_auc')
					elif self.validation.get('cross_val_score'):
						score = Splits().cross_validation(xgboost,self.trainF,self.trainL,self.validation,mission)
					elif self.validation.get('time_series_score'):
						score = Splits().time_series_validation(xgboost,self.trainF,self.trainL,self.validation)
					elif self.validation.get('group_time_series_score'):
						trainTimeID = self.trainTimeID.apply(lambda x: int(str(x)[0:6]))
						score = Splits().group_time_series_validation(xgboost,self.trainF,self.trainL,trainTimeID,self.validation)
					return -score
				
				if len(kwargs) == 0:
					n_jobs_lst = [-1]
					objective_lst = ['binary:logistic','binary:logitraw']
					space = {'n_jobs':hp.choice('n_jobs',n_jobs_lst),
							'learning_rate': hp.uniform('learning_rate',0.1,1.0),
							'n_estimators': scope.int(hp.uniform('n_estimators',200,400)),
							'max_depth': scope.int(hp.uniform('max_depth',3,6)),
							'objective':hp.choice('objective',objective_lst)}
					optparams = fmin(fn=xgboost,space=space,algo=tpe.suggest,max_evals=10)
				else:
					uniform_lst,int_lst,choice_lst = at(self.model_params['xgboost']['classification'],'uniform','int','choice')
					space,int_key,choice_key = {},[],[]
					for key,value in kwargs.items():
						if key in choice_lst or len(value) == 1:
							space.update({key:hp.choice(key,value)})
							choice_key.append((key,value))
						elif key in uniform_lst:
							space.update({key:hp.uniform(key,value[0],value[1])})
						elif key in int_lst:
							space.update({key:scope.int(hp.uniform(key,value[0],value[1]))})
							int_key.append(key)
						else:
							print('key %s does not belongs to uniform/int/choice, thus using default value instead'%(key))
					optparams = fmin(fn=xgboost,space=space,algo=tpe.suggest,max_evals=10)
					for key in int_key:
						optparams[key] = int(optparams[key])
					for item in choice_key:
						optparams.update({item[0]:item[1][optparams[item[0]]]})
				xgboost = XGBClassifier()
				xgboost.set_params(**optparams)
				xgboostTrained = xgboost.fit(self.trainF,self.trainL)

		elif mission == 'regression':
			if self.paraTuneMethod == 'NoHyperTune':
				print('No hyperparameter optimization needed, now training the model')
				xgboost = XGBRegressor(eval_metric='mlogloss',use_label_encoder=False,**kwargs)
				xgboostTrained = xgboost.fit(self.trainF,self.trainR)

			elif self.paraTuneMethod == 'gridsearchcv':
				print('use grid search method to do hyperparameter optimizaton, now training the model')
				xgboost = XGBRegressor(eval_metric='mlogloss',use_label_encoder=False)

				if len(kwargs) == 0:
					paramSpace = {'n_jobs':[-1],
									'learning_rate':[0.1,0.5,1.0], 
									'n_estimators':[200,300,400], 
									'max_depth':[3,4,5,6], 
									'objective': ['binary:logistic']} 
				else:
					paramSpace = kwargs
				xgboost = GridSearchCV(xgboost,paramSpace,cv=4,n_jobs=-1)
				xgboostTrained = xgboost.fit(self.trainF,self.trainR)

			elif self.paraTuneMethod == 'hyperopt':
				print('use bayesian hyperparameter optimization, now training the model')
				
				def xgboost(paramSpace):
					xgboost = XGBRegressor()
					xgboost.set_params(**paramSpace)
					if self.validation == None or (not self.validation.get('xgboost')):
						score = cross_val_score(xgboost,self.trainF,self.trainR,cv=4,scoring='neg_mean_squared_error').mean()
						print('use cross validation with parameters cv=4, scoring=neg_mean_squared_error')
					elif self.validation.get('cross_val_score'):
						score = Splits().cross_validation(xgboost,self.trainF,self.trainR,self.validation,mission)
					elif self.validation.get('time_series_score'):
						score = Splits().time_series_validation(xgboost,self.trainF,self.trainR,self.validation)
					elif self.validation.get('group_time_series_score'):
						trainTimeID = self.trainTimeID.apply(lambda x: int(str(x)[0:6]))
						score = Splits().group_time_series_validation(xgboost,self.trainF,self.trainR,trainTimeID,self.validation)
					return abs(score)
				
				if len(kwargs) == 0:
					n_jobs_lst = [-1]
					objective_lst = ['binary:logistic','binary:logitraw']
					space = {'n_jobs':hp.choice('n_jobs',n_jobs_lst),
							'learning_rate': hp.uniform('learning_rate',0.1,1.0),
							'n_estimators': scope.int(hp.uniform('n_estimators',200,400)),
							'max_depth': scope.int(hp.uniform('max_depth',3,6)),
							'objective':hp.choice('objective',objective_lst)}
					optparams = fmin(fn=xgboost,space=space,algo=tpe.suggest,max_evals=10)
				else:
					uniform_lst,int_lst,choice_lst = at(self.model_params['xgboost']['regression'],'uniform','int','choice')
					space,int_key,choice_key = {},[],[]
					for key,value in kwargs.items():
						if key in choice_lst or len(value) == 1:
							space.update({key:hp.choice(key,value)})
							choice_key.append((key,value))
						elif key in uniform_lst:
							space.update({key:hp.uniform(key,value[0],value[1])})
						elif key in int_lst:
							space.update({key:scope.int(hp.uniform(key,value[0],value[1]))})
							int_key.append(key)
						else:
							print('key %s does not belongs to uniform/int/choice, thus using default value instead'%(key))
					optparams = fmin(fn=xgboost,space=space,algo=tpe.suggest,max_evals=10)
					for key in int_key:
						optparams[key] = int(optparams[key])
					for item in choice_key:
						optparams.update({item[0]:item[1][optparams[item[0]]]})
				xgboost = XGBRegressor()
				xgboost.set_params(**optparams)
				xgboostTrained = xgboost.fit(self.trainF,self.trainR)
		
		else:
			raise ValueError('The mission of xgboost should be regression or classification instead of %s'%(mission))

		return xgboostTrained

	# light gradient boosting machine
	def lightgbm(self,mission='classification',**kwargs):
		if mission == 'classification':
			if self.paraTuneMethod == 'NoHyperTune':
				print('No hyperparameter optimization needed, now training the model')
				lightgbm = LGBMClassifier(**kwargs)
				lightgbmTrained = lightgbm.fit(self.trainF,self.trainL)

			elif self.paraTuneMethod == 'gridsearchcv':
				print('use grid search method to do hyperparameter optimizaton, now training the model')
				lightgbm = LGBMClassifier()

				if len(kwargs) == 0:
					paramSpace = {'learning_rate':[0.1,0.5,1.0], 
								'max_depth': range(3,8,1), 
								'min_child_weight':range(1,5,1), 
								'n_estimators':[200,300,400],
								}
				else:
					paramSpace = kwargs
				lightgbm = GridSearchCV(lightgbm,paramSpace,cv=4,n_jobs=-1)
				lightgbmTrained = lightgbm.fit(self.trainF,self.trainL)

			elif self.paraTuneMethod == 'hyperopt':
				print('use bayesian hyperparameter optimization, now training the model')
				
				def lightgbm(paramSpace):
					lightgbm = LGBMClassifier()
					lightgbm.set_params(**paramSpace)
					if self.validation == None:
						score = cross_val_score(lightgbm,self.trainF,self.trainL,cv=4,scoring='roc_auc').mean()
						print('use cross validation with parameters cv=4, scoring=roc_auc')
					elif self.validation.get('cross_val_score'):
						score = Splits().cross_validation(lightgbm,self.trainF,self.trainL,self.validation,mission)
					elif self.validation.get('time_series_score'):
						score = Splits().time_series_validation(lightgbm,self.trainF,self.trainL,self.validation)
					elif self.validation.get('group_time_series_score'):
						trainTimeID = self.trainTimeID.apply(lambda x: int(str(x)[0:6]))
						score = Splits().group_time_series_validation(lightgbm,self.trainF,self.trainL,trainTimeID,self.validation)
					return -score
					
				if len(kwargs) == 0:
					space = {'learning_rate':hp.uniform('learning_rate',0.1,1.0), 
							'max_depth': scope.int(hp.uniform('max_depth',3,8)),
							'min_child_weight':scope.int(hp.uniform('min_child_weight',1,5)),
							'n_estimators':scope.int(hp.uniform('n_estimators',200,400))}
					optparams = fmin(fn=lightgbm,space=space,algo=tpe.suggest,max_evals=10)
				else:
					uniform_lst,int_lst,choice_lst = at(self.model_params['lightgbm']['classification'],'uniform','int','choice')
					space,int_key,choice_key = {},[],[]
					for key,value in kwargs.items():
						if key in choice_lst or len(value) == 1:
							space.update({key:hp.choice(key,value)})
							choice_key.append((key,value))
						elif key in uniform_lst:
							space.update({key:hp.uniform(key,value[0],value[1])})
						elif key in int_lst:
							space.update({key:scope.int(hp.uniform(key,value[0],value[1]))})
							int_key.append(key)
						else:
							print('key %s does not belongs to uniform/int/choice, thus using default value instead'%(key))
					optparams = fmin(fn=lightgbm,space=space,algo=tpe.suggest,max_evals=10)
					for key in int_key:
						optparams[key] = int(optparams[key])
					for item in choice_key:
						optparams.update({item[0]:item[1][optparams[item[0]]]})
				lightgbm = LGBMClassifier()
				lightgbm.set_params(**optparams)
				lightgbmTrained = lightgbm.fit(self.trainF,self.trainL)

		elif mission == 'regression':
			if self.paraTuneMethod == 'NoHyperTune':
				print('No hyperparameter optimization needed, now training the model')
				lightgbm = LGBMRegressor(**kwargs)
				lightgbmTrained = lightgbm.fit(self.trainF,self.trainR)

			elif self.paraTuneMethod == 'gridsearchcv':
				print('use grid search method to do hyperparameter optimizaton, now training the model')
				lightgbm = LGBMRegressor()

				if len(kwargs) == 0:
					paramSpace = {'learning_rate':[0.1,0.5,1.0], 
								'max_depth': range(3,8,1), 
								'n_estimators':[200,300,400],
								}
				else:
					paramSpace = kwargs
				lightgbm = GridSearchCV(lightgbm,paramSpace,cv=4,n_jobs=-1)
				lightgbmTrained = lightgbm.fit(self.trainF,self.trainR)

			elif self.paraTuneMethod == 'hyperopt':
				print('use bayesian hyperparameter optimization, now training the model')
				
				def lightgbm(paramSpace):
					lightgbm = LGBMRegressor()
					lightgbm.set_params(**paramSpace)
					if self.validation == None:
						score = cross_val_score(lightgbm,self.trainF,self.trainR,cv=4,scoring='neg_mean_squared_error').mean()
						print('use cross validation with parameters cv=4, scoring=neg_mean_squared_error')
					elif self.validation.get('cross_val_score'):
						score = Splits().cross_validation(lightgbm,self.trainF,self.trainR,self.validation,mission)
					elif self.validation.get('time_series_score'):
						score = Splits().time_series_validation(lightgbm,self.trainF,self.trainR,self.validation)
					elif self.validation.get('group_time_series_score'):
						trainTimeID = self.trainTimeID.apply(lambda x: int(str(x)[0:6]))
						score = Splits().group_time_series_validation(xgboost,self.trainF,self.trainR,trainTimeID,self.validation)
					return abs(score)
					
				if len(kwargs) == 0:
					space = {'learning_rate':hp.uniform('learning_rate',0.1,1.0), 
							'max_depth': scope.int(hp.uniform('max_depth',3,8)),
							'n_estimators':scope.int(hp.uniform('n_estimators',200,400))}
					optparams = fmin(fn=lightgbm,space=space,algo=tpe.suggest,max_evals=10)
				else:
					uniform_lst,int_lst,choice_lst = at(self.model_params['lightgbm']['regression'],'uniform','int','choice')
					space,int_key,choice_key = {},[],[]
					for key,value in kwargs.items():
						if key in choice_lst or len(value) == 1:
							space.update({key:hp.choice(key,value)})
							choice_key.append((key,value))
						elif key in uniform_lst:
							space.update({key:hp.uniform(key,value[0],value[1])})
						elif key in int_lst:
							space.update({key:scope.int(hp.uniform(key,value[0],value[1]))})
							int_key.append(key)
						else:
							print('key %s does not belongs to uniform/int/choice, thus using default value instead'%(key))
					optparams = fmin(fn=lightgbm,space=space,algo=tpe.suggest,max_evals=10)
					for key in int_key:
						optparams[key] = int(optparams[key])
					for item in choice_key:
						optparams.update({item[0]:item[1][optparams[item[0]]]})
				lightgbm = LGBMRegressor()
				lightgbm.set_params(**optparams)
				lightgbmTrained = lightgbm.fit(self.trainF,self.trainR)
		else:
			raise ValueError('The mission of light gradient boosting machine should be regression or classification instead of %s'%(mission))

		return lightgbmTrained

	# catboost
	def catboost(self,mission='classification',**kwargs):
		if mission == 'classification':
			if self.paraTuneMethod == 'NoHyperTune':
				print('No hyperparameter optimization needed, now training the model')
				catboost = CatBoostClassifier(**kwargs)
				catboostTrained = catboost.fit(self.trainF,self.trainL)

			elif self.paraTuneMethod == 'gridsearchcv':
				print('use grid search method to do hyperparameter optimizaton, now training the model')
				catboost = CatBoostClassifier()
				catboostTrained = catboost.fit(self.trainF,self.trainL)

				if len(kwargs) == 0:
					paramSpace = {'learning_rate':[0.1,0.5,1.0],
								'max_depth': range(3,8,1),
								'n_estimators':[200,300,400],
								}
				else:
					paramSpace = kwargs
				catboost = GridSearchCV(catboost,paramSpace,cv=4,n_jobs=-1)

			elif self.paraTuneMethod == 'hyperopt':
				print('use bayesian hyperparameter optimization, now training the model')
				
				def catboost(paramSpace):
					catboost = CatBoostClassifier()
					catboost.set_params(**paramSpace)
					if self.validation == None or (not self.validation.get('catboost')):
						score = cross_val_score(catboost,self.trainF,self.trainL,cv=4,scoring='roc_auc').mean()
						print('use cross validation with parameters cv=4, scoring=roc_auc')
					elif self.validation.get('cross_val_score'):
						score = Splits().cross_validation(catboost,self.trainF,self.trainL,self.validation,mission)
					elif self.validation.get('time_series_score'):
						score = Splits().time_series_validation(catboost,self.trainF,self.trainL,self.validation)
					elif self.validation.get('group_time_series_score'):
						trainTimeID = self.trainTimeID.apply(lambda x: int(str(x)[0:6]))
						score = Splits().group_time_series_validation(catboost,self.trainF,self.trainL,trainTimeID,self.validation)
					return -score
					
				if len(kwargs) == 0:
					space = {'learning_rate':hp.uniform('learning_rate',0.1,1.0), 
							'max_depth': scope.int(hp.uniform('max_depth',3,8)),
							'n_estimators':scope.int(hp.uniform('n_estimators',200,400))}
					optparams = fmin(fn=catboost,space=space,algo=tpe.suggest,max_evals=10)
				else:
					uniform_lst,int_lst,choice_lst = at(self.model_params['catboost']['classification'],'uniform','int','choice')
					space,int_key,choice_key = {},[],[]
					for key,value in kwargs.items():
						if key in choice_lst or len(value) == 1:
							space.update({key:hp.choice(key,value)})
							choice_key.append((key,value))
						elif key in uniform_lst:
							space.update({key:hp.uniform(key,value[0],value[1])})
						elif key in int_lst:
							space.update({key:scope.int(hp.uniform(key,value[0],value[1]))})
							int_key.append(key)
						else:
							print('key %s does not belongs to uniform/int/choice, thus using default value instead'%(key))
					optparams = fmin(fn=catboost,space=space,algo=tpe.suggest,max_evals=10)
					for key in int_key:
						optparams[key] = int(optparams[key])
					for item in choice_key:
						optparams.update({item[0]:item[1][optparams[item[0]]]})
				catboost = CatBoostClassifier()
				catboost.set_params(**optparams)
				catboostTrained = catboost.fit(self.trainF,self.trainL)

		elif mission == 'regression':
			if self.paraTuneMethod == 'NoHyperTune':
				print('No hyperparameter optimization needed, now training the model')
				catboost = CatBoostRegressor(**kwargs)
				catboostTrained = catboost.fit(self.trainF,self.trainR)

			elif self.paraTuneMethod == 'gridsearchcv':
				print('use grid search method to do hyperparameter optimizaton, now training the model')
				catboost = CatBoostRegressor()
				catboostTrained = catboost.fit(self.trainF,self.trainR)

				if len(kwargs) == 0:
					paramSpace = {'learning_rate':[0.1,0.5,1.0],
								'max_depth': range(3,8,1),
								'n_estimators':[200,300,400],
								}
				else:
					paramSpace = kwargs
				catboost = GridSearchCV(catboost,paramSpace,cv=4,n_jobs=-1)

			elif self.paraTuneMethod == 'hyperopt':
				print('use bayesian hyperparameter optimization, now training the model')
				
				def catboost(paramSpace):
					catboost = CatBoostRegressor()
					catboost.set_params(**paramSpace)
					if self.validation == None or (not self.validation.get('catboost')):
						score = cross_val_score(catboost,self.trainF,self.trainR,cv=4,scoring='neg_mean_squared_error').mean()
						print('use cross validation with parameters cv=4, scoring=neg_mean_squared_error')
					elif self.validation.get('cross_val_score'):
						score = Splits().cross_validation(catboost,self.trainF,self.trainR,self.validation,mission)
					elif self.validation.get('time_series_score'):
						score = Splits().time_series_validation(catboost,self.trainF,self.trainR,self.validation)
					elif self.validation.get('group_time_series_score'):
						trainTimeID = self.trainTimeID.apply(lambda x: int(str(x)[0:6]))
						score = Splits().group_time_series_validation(catboost,self.trainF,self.trainR,trainTimeID,self.validation)
					return abs(score)

				if len(kwargs) == 0:
					space = {'learning_rate':hp.uniform('learning_rate',0.1,1.0), 
							'max_depth': scope.int(hp.uniform('max_depth',3,8)),
							'n_estimators':scope.int(hp.uniform('n_estimators',200,400))}
					optparams = fmin(fn=catboost,space=space,algo=tpe.suggest,max_evals=10)
				else:
					uniform_lst,int_lst,choice_lst = at(self.model_params['catboost']['regression'],'uniform','int','choice')
					space,int_key,choice_key = {},[],[]
					for key,value in kwargs.items():
						if key in choice_lst or len(value) == 1:
							space.update({key:hp.choice(key,value)})
							choice_key.append((key,value))
						elif key in uniform_lst:
							space.update({key:hp.uniform(key,value[0],value[1])})
						elif key in int_lst:
							space.update({key:scope.int(hp.uniform(key,value[0],value[1]))})
							int_key.append(key)
						else:
							print('key %s does not belongs to uniform/int/choice, thus using default value instead'%(key))
					optparams = fmin(fn=catboost,space=space,algo=tpe.suggest,max_evals=10)
					for key in int_key:
						optparams[key] = int(optparams[key])
					for item in choice_key:
						optparams.update({item[0]:item[1][optparams[item[0]]]})
				catboost = CatBoostRegressor()
				catboost.set_params(**optparams)
				catboostTrained = catboost.fit(self.trainF,self.trainR)
		else:
			raise ValueError('The mission of catboost should be regression or classification instead of %s'%(mission))

		return catboostTrained

	# logistic regression
	def logistic(self,mission='classification',**kwargs):
		if mission == 'classification':
			if self.paraTuneMethod == 'NoHyperTune':
				print('No hyperparameter optimization needed, now training the model')
				logistic = LogisticRegression(**kwargs)
				logisticTrained = logistic.fit(self.trainF,self.trainL)

			elif self.paraTuneMethod == 'gridsearchcv':
				print('use grid search method to do hyperparameter optimizaton, now training the model')
				logistic = LogisticRegression()	
				if len(kwargs) == 0:
					paramSpace = {'penalty':['l1','l2'], 
								'C': [0.001, 0.01, 0.1, 1, 10]} 
				else:
					paramSpace = kwargs
				logistic = GridSearchCV(logistic,paramSpace,cv=4,n_jobs=-1)
				logisticTrained = logistic.fit(self.trainF,self.trainL)

			elif self.paraTuneMethod == 'hyperopt':
				print('use bayesian hyperparameter optimization, now training the model')
				
				def logistic(paramSpace):
					logistic = LogisticRegression()
					logistic.set_params(**paramSpace)
					if self.validation == None:
						score = cross_val_score(logistic,self.trainF,self.trainL,cv=4,scoring='roc_auc').mean()
						print('use cross validation with parameters cv=4, scoring=roc_auc')
					elif self.validation.get('cross_val_score'):
						score = Splits().cross_validation(logistic,self.trainF,self.trainL,self.validation,mission)
					elif self.validation.get('time_series_score'):
						score = Splits().time_series_validation(logistic,self.trainF,self.trainL,self.validation)
					elif self.validation.get('group_time_series_score'):
						trainTimeID = self.trainTimeID.apply(lambda x: int(str(x)[0:6]))
						score = Splits().group_time_series_validation(logistic,self.trainF,self.trainL,trainTimeID,self.validation)
					return -score

				if len(kwargs) == 0:
					penalty_lst = ['l2']
					space = {'penalty':hp.choice('penalty',penalty_lst),
							'C':hp.uniform('C',0.001,10)}
					optparams = fmin(fn=logistic,space=space,algo=tpe.suggest,max_evals=10)
				else:
					uniform_lst,int_lst,choice_lst = at(self.model_params['logistic'],'uniform','int','choice')
					space,int_key,choice_key = {},[],[]
					for key,value in kwargs.items():
						if key in choice_lst or len(value) == 1:
							space.update({key:hp.choice(key,value)})
							choice_key.append((key,value))
						elif key in uniform_lst:
							space.update({key:hp.uniform(key,value[0],value[1])})
						elif key in int_lst:
							space.update({key:scope.int(hp.uniform(key,value[0],value[1]))})
							int_key.append(key)
						else:
							print('key %s does not belongs to uniform/int/choice, thus using default value instead'%(key))
					optparams = fmin(fn=logistic,space=space,algo=tpe.suggest,max_evals=10)
					for key in int_key:
						optparams[key] = int(optparams[key])
					for item in choice_key:
						optparams.update({item[0]:item[1][optparams[item[0]]]})
				logistic = LogisticRegression()
				optparams.update({'penalty':penalty_lst[optparams['penalty']]})
				logistic.set_params(**optparams)
				logisticTrained = logistic.fit(self.trainF,self.trainL)
		else:
			raise ValueError('The mission of logistic regression should be classification instead of %s'%(mission))

		return logisticTrained

	# stochastic gradient descent
	def sgd(self,mission='classification',**kwargs):
		if mission == 'classification':
			if self.paraTuneMethod == 'NoHyperTune':
				print('No hyperparameter optimization needed, now training the model')
				sgd = SGDClassifier(**kwargs)
				sgdTrained = sgd.fit(self.trainF,self.trainL)

			elif self.paraTuneMethod == 'gridsearchcv':
				print('use grid search method to do hyperparameter optimizaton, now training the model')
				sgd = SGDClassifier()	
				if len(kwargs) == 0:
					paramSpace = {'alpha': [0.1, 0.01, 0.001, 0.0001],
								'penalty':['l1','l2']}
				else:
					paramSpace = kwargs
				sgd = GridSearchCV(sgd,paramSpace,cv=4,n_jobs=-1)
				sgdTrained = sgd.fit(self.trainF,self.trainL)

			elif self.paraTuneMethod == 'hyperopt':
				print('use bayesian hyperparameter optimization, now training the model')
				
				def sgd(paramSpace):
					sgd = SGDClassifier()
					sgd.set_params(**paramSpace)
					if self.validation == None:
						score = cross_val_score(sgd,self.trainF,self.trainL,cv=4,scoring='roc_auc').mean()
						print('use cross validation with parameters cv=4, scoring=roc_auc')
					elif self.validation.get('cross_val_score'):
						score = Splits().cross_validation(sgd,self.trainF,self.trainL,self.validation,mission)
					elif self.validation.get('time_series_score'):
						score = Splits().time_series_validation(sgd,self.trainF,self.trainL,self.validation)
					elif self.validation.get('group_time_series_score'):
						trainTimeID = self.trainTimeID.apply(lambda x: int(str(x)[0:6]))
						score = Splits().group_time_series_validation(sgd,self.trainF,self.trainL,trainTimeID,self.validation)
					return -score
					
				if len(kwargs) == 0:
					penalty_lst = ['l1','l2']
					space = {'penalty':hp.choice('penalty',penalty_lst),
							'alpha':hp.uniform('alpha',0.001,10)}
					optparams = fmin(fn=sgd,space=space,algo=tpe.suggest,max_evals=10)
				else:
					uniform_lst,int_lst,choice_lst = at(self.model_params['sgd']['classification'],'uniform','int','choice')
					space,int_key,choice_key = {},[],[]
					for key,value in kwargs.items():
						if key in choice_lst or len(value) == 1:
							space.update({key:hp.choice(key,value)})
							choice_key.append((key,value))
						elif key in uniform_lst:
							space.update({key:hp.uniform(key,value[0],value[1])})
						elif key in int_lst:
							space.update({key:scope.int(hp.uniform(key,value[0],value[1]))})
							int_key.append(key)
						else:
							print('key %s does not belongs to uniform/int/choice, thus using default value instead'%(key))
					optparams = fmin(fn=sgd,space=space,algo=tpe.suggest,max_evals=10)
					for key in int_key:
						optparams[key] = int(optparams[key])
					for item in choice_key:
						optparams.update({item[0]:item[1][optparams[item[0]]]})
				sgd = SGDClassifier()
				sgd.set_params(**optparams)
				sgdTrained = sgd.fit(self.trainF,self.trainL)
		
		elif mission == 'regression':
			if self.paraTuneMethod == 'NoHyperTune':
				print('No hyperparameter optimization needed, now training the model')
				sgd = SGDRegressor(**kwargs)
				sgdTrained = sgd.fit(self.trainF,self.trainR)

			elif self.paraTuneMethod == 'gridsearchcv':
				print('use grid search method to do hyperparameter optimizaton, now training the model')
				sgd = SGDRegressor()	
				if len(kwargs) == 0:
					paramSpace = {'alpha': [0.1, 0.01, 0.001, 0.0001], 
								'penalty': ['l1','l2']} 
				else:
					paramSpace = kwargs
				sgd = GridSearchCV(sgd,paramSpace,cv=4,n_jobs=-1)
				sgdTrained = sgd.fit(self.trainF,self.trainR)

			elif self.paraTuneMethod == 'hyperopt':
				print('use bayesian hyperparameter optimization, now training the model')
				
				def sgd(paramSpace):
					sgd = SGDRegressor()
					sgd.set_params(**paramSpace)
					if self.validation == None:
						score = cross_val_score(sgd,self.trainF,self.trainR,cv=4,scoring='neg_mean_squared_error').mean()
						print('use cross validation with parameters cv=4, scoring=neg_mean_squared_error')
					elif self.validation.get('cross_val_score'):
						score = Splits().cross_validation(sgd,self.trainF,self.trainR,self.validation,mission)
					elif self.validation.get('time_series_score'):
						score = Splits().time_series_validation(sgd,self.trainF,self.trainR,self.validation)
					elif self.validation.get('group_time_series_score'):
						trainTimeID = self.trainTimeID.apply(lambda x: int(str(x)[0:6]))
						score = Splits().group_time_series_validation(sgd,self.trainF,self.trainR,trainTimeID,self.validation)
					return abs(score)
					
				if len(kwargs) == 0:
					penalty_lst = ['l1','l2']
					space = {'penalty':hp.choice('penalty',penalty_lst),
							'alpha':hp.uniform('alpha',0.001,10)}
					optparams = fmin(fn=sgd,space=space,algo=tpe.suggest,max_evals=10)
				else:
					uniform_lst,int_lst,choice_lst = at(self.model_params['sgd']['regression'],'uniform','int','choice')
					space,int_key,choice_key = {},[],[]
					for key,value in kwargs.items():
						if key in choice_lst or len(value) == 1:
							space.update({key:hp.choice(key,value)})
							choice_key.append((key,value))
						elif key in uniform_lst:
							space.update({key:hp.uniform(key,value[0],value[1])})
						elif key in int_lst:
							space.update({key:scope.int(hp.uniform(key,value[0],value[1]))})
							int_key.append(key)
						else:
							print('key %s does not belongs to uniform/int/choice, thus using default value instead'%(key))
					optparams = fmin(fn=sgd,space=space,algo=tpe.suggest,max_evals=10)
					for key in int_key:
						optparams[key] = int(optparams[key])
					for item in choice_key:
						optparams.update({item[0]:item[1][optparams[item[0]]]})
				sgd = SGDRegressor()
				sgd.set_params(**optparams)
				sgdTrained = sgd.fit(self.trainF,self.trainR)
		else:
			raise ValueError('The mission of stochastic gradient descent should be regression or classification instead of %s'%(mission))

		return sgdTrained

	# linear support vector classification
	def linearSVC(self,mission='classification',**kwargs):
		if mission == 'classification':
			if self.paraTuneMethod == 'NoHyperTune':
				print('No hyperparameter optimization needed, now training the model')
				linearSVC = LinearSVC(**kwargs)
				linearSVCTrained = linearSVC.fit(self.trainF,self.trainL)

			elif self.paraTuneMethod == 'gridsearchcv':
				print('use grid search method to do hyperparameter optimizaton, now training the model')
				linearSVC = LinearSVC()	
				if len(kwargs) == 0:
					paramSpace = {'C': [0.001, 0.01, 0.1, 1, 10]} 
				else:
					paramSpace = kwargs
				linearSVC = GridSearchCV(linearSVC,paramSpace,cv=4,n_jobs=-1)
				linearSVCTrained = linearSVC.fit(self.trainF,self.trainL)

			elif self.paraTuneMethod == 'hyperopt':
				print('use bayesian hyperparameter optimization, now training the model')
				
				def linearSVC(paramSpace):
					linearSVC = LinearSVC()
					linearSVC.set_params(**paramSpace)
					if self.validation == None:
						score = cross_val_score(linearSVC,self.trainF,self.trainL,cv=4,scoring='roc_auc').mean()
						print('use cross validation with parameters cv=4, scoring=roc_auc')
					elif self.validation.get('cross_val_score'):
						score = Splits().cross_validation(linearSVC,self.trainF,self.trainL,self.validation,mission)
					elif self.validation.get('time_series_score'):
						score = Splits().time_series_validation(linearSVC,self.trainF,self.trainL,self.validation)
					elif self.validation.get('group_time_series_score'):
						trainTimeID = self.trainTimeID.apply(lambda x: int(str(x)[0:6]))
						score = Splits().group_time_series_validation(linearSVC,self.trainF,self.trainL,trainTimeID,self.validation)
					return -score
					
				if len(kwargs) == 0:
					space = {'C':hp.uniform('C',0.001,10)}
					optparams = fmin(fn=linearSVC,space=space,algo=tpe.suggest,max_evals=10)
				else:
					uniform_lst,int_lst,choice_lst = at(self.model_params['linearsvc'],'uniform','int','choice')
					space,int_key,choice_key = {},[],[]
					for key,value in kwargs.items():
						if key in choice_lst or len(value) == 1:
							space.update({key:hp.choice(key,value)})
							choice_key.append((key,value))
						elif key in uniform_lst:
							space.update({key:hp.uniform(key,value[0],value[1])})
						elif key in int_lst:
							space.update({key:scope.int(hp.uniform(key,value[0],value[1]))})
							int_key.append(key)
						else:
							print('key %s does not belongs to uniform/int/choice, thus using default value instead'%(key))
					optparams = fmin(fn=linearSVC,space=space,algo=tpe.suggest,max_evals=10)
					for key in int_key:
						optparams[key] = int(optparams[key])
					for item in choice_key:
						optparams.update({item[0]:item[1][optparams[item[0]]]})
				linearSVC = LinearSVC()
				linearSVC.set_params(**optparams)
				linearSVCTrained = linearSVC.fit(self.trainF,self.trainL)
		else:
			raise ValueError('The mission of linear support vector classification should be classification instead of %s'%(mission))

		return linearSVCTrained

	# support vector machine
	def svm(self,mission='classification',**kwargs):
		if mission == 'classification':
			if self.paraTuneMethod == 'NoHyperTune':
				print('No hyperparameter optimization needed, now training the model')
				svm = SVC(**kwargs)
				svmTrained = svm.fit(self.trainF,self.trainL)

			elif self.paraTuneMethod == 'gridsearchcv':
				print('use grid search method to do hyperparameter optimizaton, now training the model')
				svm = SVC()	
				if len(kwargs) == 0:
					paramSpace = {'kernel': ['rbf'], 
								'gamma': [0.01,0.1,1,10], 
								'C': [0.001,0.01,0.1,1,10]} 
				else:
					paramSpace = kwargs
				svm = GridSearchCV(svm,paramSpace,cv=4,n_jobs=-1)
				svmTrained = svm.fit(self.trainF,self.trainL)

			elif self.paraTuneMethod == 'hyperopt':
				print('use bayesian hyperparameter optimization, now training the model')
				
				def svm(paramSpace):
					svm = SVC()
					svm.set_params(**paramSpace)
					if self.validation == None:
						score = cross_val_score(svm,self.trainF,self.trainL,cv=4,scoring='roc_auc').mean()
						print('use cross validation with parameters cv=4, scoring=roc_auc')
					elif self.validation.get('cross_val_score'):
						score = Splits().cross_validation(svm,self.trainF,self.trainL,self.validation,mission)
					elif self.validation.get('time_series_score'):
						score = Splits().time_series_validation(svm,self.trainF,self.trainL,self.validation)
					elif self.validation.get('group_time_series_score'):
						trainTimeID = self.trainTimeID.apply(lambda x: int(str(x)[0:6]))
						score = Splits().group_time_series_validation(svm,self.trainF,self.trainL,trainTimeID,self.validation)
					return -score
					
				if len(kwargs) == 0:
					kernel_lst = ['linear','rbf','sigmoid']	
					space = {'kernel':hp.choice('kernel',kernel_lst),
							'gamma':hp.uniform('gamma',0.01,10),
							'C':hp.uniform('C',0.001,10)}
					optparams = fmin(fn=svm,space=space,algo=tpe.suggest,max_evals=10)
				else:
					uniform_lst,int_lst,choice_lst = at(self.model_params['svm'],'uniform','int','choice')
					space,int_key,choice_key = {},[],[]
					for key,value in kwargs.items():
						if key in choice_lst or len(value) == 1:
							space.update({key:hp.choice(key,value)})
							choice_key.append((key,value))
						elif key in uniform_lst:
							space.update({key:hp.uniform(key,value[0],value[1])})
						elif key in int_lst:
							space.update({key:scope.int(hp.uniform(key,value[0],value[1]))})
							int_key.append(key)
						else:
							print('key %s does not belongs to uniform/int/choice, thus using default value instead'%(key))
					optparams = fmin(fn=svm,space=space,algo=tpe.suggest,max_evals=10)
					for key in int_key:
						optparams[key] = int(optparams[key])
					for item in choice_key:
						optparams.update({item[0]:item[1][optparams[item[0]]]})
				svm = SVC()
				svm.set_params(**optparams)
				svmTrained = svm.fit(self.trainF,self.trainL)
		else:
			raise ValueError('The mission of support vector machine should be classification instead of %s'%(mission))

		return svmTrained

	# deep forest
	def deepforest(self,mission='regression',**kwargs):
		if mission == 'regression':
			self.trainF,self.trainR = self.trainF.values,self.trainR.values
			if self.paraTuneMethod == 'NoHyperTune':
				print('No hyperparameter optimization needed, now training the model')
				deepforest = CascadeForestRegressor(**kwargs)
				deepforest.fit(self.trainF,self.trainR)

			elif self.paraTuneMethod == 'gridsearchcv':
				print('use grid search method to do hyperparameter optimizaton, now training the model')
				deepforest = CascadeForestRegressor()	

				if len(kwargs) == 0:
					paramSpace = {'n_trees': [100, 200, 300, 400],
								  'max_depth': [2,3,4,5]
								}
				else:
					paramSpace = kwargs
				deepforest = GridSearchCV(deepforest,paramSpace,cv=4,n_jobs=-1)
				deepforest.fit(self.trainF,self.trainR)

			elif self.paraTuneMethod == 'hyperopt':
				print('use bayesian hyperparameter optimization, now training the model')

				def deepforest(paramSpace):
					deepforest = CascadeForestRegressor()
					deepforest.set_params(**paramSpace)
					if self.validation == None:
						score = cross_val_score(deepforest,self.trainF,self.trainR,cv=4,scoring='neg_mean_squared_error').mean()
						print('use cross validation with parameters cv=4, scoring=neg_mean_squared_error')
					elif self.validation.get('cross_val_score'):
						score = Splits().cross_validation(deepforest,self.trainF,self.trainR,self.validation,mission)
					elif self.validation.get('time_series_score'):
						score = Splits().time_series_validation(deepforest,self.trainF,self.trainR,self.validation)
					elif self.validation.get('group_time_series_score'):
						trainTimeID = self.trainTimeID.apply(lambda x: int(str(x)[0:6]))
						score = Splits().group_time_series_validation(deepforest,self.trainF,self.trainR,trainTimeID,self.validation)
					return abs(score)

				if len(kwargs) == 0:
					space = {'n_trees': scope.int(hp.uniform('n_trees',100,400)),
							 'max_depth': scope.int(hp.uniform('max_depth',2,5))}
					optparams = fmin(fn=deepforest,space=space,algo=tpe.suggest,max_evals=10)
				else:
					uniform_lst,int_lst,choice_lst = at(self.model_params['deepforest']['regression'],'uniform','int','choice')
					space,int_key,choice_key = {},[],[]
					for key,value in kwargs.items():
						if key in choice_lst or len(value) == 1:
							space.update({key:hp.choice(key,value)})
							choice_key.append((key,value))
						elif key in uniform_lst:
							space.update({key:hp.uniform(key,value[0],value[1])})
						elif key in int_lst:
							space.update({key:scope.int(hp.uniform(key,value[0],value[1]))})
							int_key.append(key)
						else:
							print('key %s does not belongs to uniform/int/choice, thus using default value instead'%(key))
					optparams = fmin(fn=deepforest,space=space,algo=tpe.suggest,max_evals=10)
					for key in int_key:
						optparams[key] = int(optparams[key])
					for item in choice_key:
						optparams.update({item[0]:item[1][optparams[item[0]]]})
				deepforest = CascadeForestRegressor()
				deepforest.set_params(**optparams)
				deepforest.fit(self.trainF,self.trainR)

		elif mission == 'classification':
			self.trainF,self.trainL = self.trainF.values,self.trainL.values
			if self.paraTuneMethod == 'NoHyperTune':
				print('No hyperparameter optimization needed, now training the model')
				deepforest = CascadeForestClassifier(**kwargs)
				deepforest.fit(self.trainF,self.trainL)

			elif self.paraTuneMethod == 'gridsearchcv':
				print('use grid search method to do hyperparameter optimizaton, now training the model')
				deepforest = CascadeForestClassifier()

				if len(kwargs) == 0:
					paramSpace = {'n_trees': [100, 200, 300, 400],
								  'max_depth': [2,3,4,5]
								}
				else:
					paramSpace = kwargs
				deepforest = GridSearchCV(deepforest,paramSpace,cv=4,n_jobs=-1)
				deepforest.fit(self.trainF,self.trainL)

			elif self.paraTuneMethod == 'hyperopt':
				print('use bayesian hyperparameter optimization, now training the model')

				def deepforest(paramSpace):
					deepforest = CascadeForestClassifier()
					deepforest.set_params(**paramSpace)
					if self.validation == None:
						score = cross_val_score(deepforest,self.trainF,self.trainL,cv=4,scoring='roc_auc').mean()
						print('use cross validation with parameters cv=4, scoring=roc_auc')
					elif self.validation.get('cross_val_score'):
						score = Splits().cross_validation(deepforest,self.trainF,self.trainL,self.validation,mission)
					elif self.validation.get('time_series_score'):
						score = Splits().time_series_validation(deepforest,self.trainF,self.trainL,self.validation)
					elif self.validation.get('group_time_series_score'):
						trainTimeID = self.trainTimeID.apply(lambda x: int(str(x)[0:6]))
						score = Splits().group_time_series_validation(deepforest,self.trainF,self.trainL,trainTimeID,self.validation)
					return -score

				if len(kwargs) == 0:
					space = {'n_trees': scope.int(hp.uniform('n_trees',100,400)),
							 'max_depth': scope.int(hp.uniform('max_depth',2,5))}
					optparams = fmin(fn=deepforest,space=space,algo=tpe.suggest,max_evals=10)
				else:
					uniform_lst,int_lst,choice_lst = at(self.model_params['deepforest']['classification'],'uniform','int','choice')
					space,int_key,choice_key = {},[],[]
					for key,value in kwargs.items():
						if key in choice_lst or len(value) == 1:
							space.update({key:hp.choice(key,value)})
							choice_key.append((key,value))
						elif key in uniform_lst:
							space.update({key:hp.uniform(key,value[0],value[1])})
						elif key in int_lst:
							space.update({key:scope.int(hp.uniform(key,value[0],value[1]))})
							int_key.append(key)
						else:
							print('key %s does not belongs to uniform/int/choice, thus using default value instead'%(key))
					optparams = fmin(fn=deepforest,space=space,algo=tpe.suggest,max_evals=10)
					for key in int_key:
						optparams[key] = int(optparams[key])
					for item in choice_key:
						optparams.update({item[0]:item[1][optparams[item[0]]]})
				deepforest = CascadeForestClassifier()
				deepforest.set_params(**optparams)
				deepforest.fit(self.trainF,self.trainL)

		return deepforest


# training selected models
class modelTrainer(object):

	def __init__(self,paraTuneMethod,trainF,trainL,trainR,testF,testL,testR,trainTimeID,trainStockID):
		self.trainF = trainF
		self.trainL = trainL
		self.trainR = trainR
		self.testF = testF
		self.testL = testL
		self.testR = testR
		self.trainTimeID = trainTimeID
		self.trainStockID = trainStockID
		self.paraTuneMethod = paraTuneMethod
		#self.metricSet = {'rankIC','sharpRatio','infoRatio','CalmarRatio'}	#the name set of running metrics 

	@funcTimer
	def kernelRidge(self,mission='regression',validation=(),**kwargs):
		# metrics = self.metricSet.intersection(kwargs)
		# a = [kwargs.pop(i) for i in metrics]	#delet the metrics then pass to model 
		# for i in metrics：

		# if 
		kernelRidgeTrained = modelSelector(self.paraTuneMethod,self.trainF,self.trainL,self.trainR,self.testF,self.testL,self.testR,
			self.trainTimeID,self.trainStockID,validation).kernelRidge(mission=mission,**kwargs)	
		# fit model and return the squared R
		if mission != 'regression':
			raise ValueError('The mission of kernel ridge regression should be regression instead of %s'%(mission))
		score = kernelRidgeTrained.score(self.testF,self.testR)
		return kernelRidgeTrained,score

	@funcTimer
	def ridge(self,mission='regression',validation=(),**kwargs):
		ridgeTrained = modelSelector(self.paraTuneMethod,self.trainF,self.trainL,self.trainR,self.testF,self.testL,self.testR,
			self.trainTimeID,self.trainStockID,validation).ridge(mission=mission,**kwargs)	
		# fit model and return the squared R
		if mission == 'regression':
			score = ridgeTrained.score(self.testF,self.testR)
		elif mission == 'classification':
			score = ridgeTrained.score(self.testF,self.testL)
		else:
			raise ValueError('The mission of ridge should be regression or classification instead of %s'%(mission))
		return ridgeTrained,score
	
	@funcTimer
	def lasso(self,mission='regression',validation=(),**kwargs):
		lassoTrained = modelSelector(self.paraTuneMethod,self.trainF,self.trainL,self.trainR,self.testF,self.testL,self.testR,
			self.trainTimeID,self.trainStockID,validation).lasso(mission=mission,**kwargs)	
		# fit model and return the squared R
		if mission != 'regression':
			raise ValueError('The mission of lasso regression should be regression instead of %s'%(mission))
		score = lassoTrained.score(self.testF,self.testR)
		return lassoTrained,score

	@funcTimer
	def elasticNet(self,mission='regression',validation=(),**kwargs):
		elasticNetTrained = modelSelector(self.paraTuneMethod,self.trainF,self.trainL,self.trainR,self.testF,self.testL,self.testR,
			self.trainTimeID,self.trainStockID,validation).elasticNet(mission=mission,**kwargs)	
		# fit model and return the squared R
		if mission != 'regression':
			raise ValueError('The mission of elastic net should be regression instead of %s'%(mission))
		score = elasticNetTrained.score(self.testF,self.testR)
		return elasticNetTrained,score
	
	@funcTimer
	def svr(self,mission='regression',validation=(),**kwargs):
		svrTrained = modelSelector(self.paraTuneMethod,self.trainF,self.trainL,self.trainR,self.testF,self.testL,self.testR,
			self.trainTimeID,self.trainStockID,validation).svr(mission=mission,**kwargs)	
		# fit model and return the squared R
		if mission != 'regression':
			raise ValueError('The mission of support vector regression should be regression instead of %s'%(mission))
		score = svrTrained.score(self.testF,self.testR)
		return svrTrained,score

	@funcTimer
	def randomForest(self,mission='classification',validation=(),**kwargs):
		randomForestTrained = modelSelector(self.paraTuneMethod,self.trainF,self.trainL,self.trainR,self.testF,self.testL,self.testR,
			self.trainTimeID,self.trainStockID,validation).randomForest(mission=mission,**kwargs)
		# fit and return the mean accuracy
		if mission == 'classification':
			score = randomForestTrained.score(self.testF,self.testL)
		elif mission == 'regression':
			score = randomForestTrained.score(self.testF,self.testR)
		else:
			raise ValueError('The mission of random forest should be regression or classification instead of %s'%(mission))
		return randomForestTrained,score
	
	@funcTimer
	def gbdt(self,mission='classification',validation=(),**kwargs):
		gbdtTrained = modelSelector(self.paraTuneMethod,self.trainF,self.trainL,self.trainR,self.testF,self.testL,self.testR,
			self.trainTimeID,self.trainStockID,validation).gbdt(mission=mission,**kwargs)
		# fit and return the mean accuracy
		if mission == 'regression':
			score = gbdtTrained.score(self.testF,self.testR)
		elif mission == 'classification':
			score = gbdtTrained.score(self.testF,self.testL)
		else:
			raise ValueError('The mission of gradient boosting decision tree should be regression or classification instead of %s'%(mission))
		return gbdtTrained,score

	@funcTimer
	def mlpc(self,mission='classification',validation=(),**kwargs):
		mlpcTrained = modelSelector(self.paraTuneMethod,self.trainF,self.trainL,self.trainR,self.testF,self.testL,self.testR,
			self.trainTimeID,self.trainStockID,validation).mlpc(mission=mission,**kwargs)
		# fit and return the mean accuracy
		if mission == 'regression':
			score = mlpcTrained.score(self.testF,self.testR)
		elif mission == 'classification':
			score = mlpcTrained.score(self.testF,self.testL)
		else:
			raise ValueError('The mission of multilayer perceptron should be regression or classification instead of %s'%(mission))
		return mlpcTrained,score

	@funcTimer
	def gnb(self,mission='classification',validation=(),**kwargs):
		gnbTrained = modelSelector(self.paraTuneMethod,self.trainF,self.trainL,self.trainR,self.testF,self.testL,self.testR,
			self.trainTimeID,self.trainStockID,validation).gnb(mission=mission,**kwargs)
		# fit and return the mean accuracy
		if mission != 'classification':
			raise ValueError('The mission of gaussian naive bayes should be classification instead of %s'%(mission))
		score = gnbTrained.score(self.testF,self.testL)
		return gnbTrained,score

	@funcTimer
	def adaboost(self,mission='classification',validation=(),**kwargs):
		adaTrained = modelSelector(self.paraTuneMethod,self.trainF,self.trainL,self.trainR,self.testF,self.testL,self.testR,
			self.trainTimeID,self.trainStockID,validation).adaboost(mission=mission,**kwargs)	
		#adaboost = super(preSetModels,self).adaboost(**kwargs)
		# fit and return the mean accuracy
		if mission == 'regression':
			score = adaTrained.score(self.testF,self.testR)
		elif mission == 'classification':
			score = adaTrained.score(self.testF,self.testL)
		else:
			raise ValueError('The mission of adaboost should be regression or classification instead of %s'%(mission))
		return adaTrained,score

	@funcTimer
	def xgboost(self,mission='classification',validation=(),**kwargs):
		xgbTrained = modelSelector(self.paraTuneMethod,self.trainF,self.trainL,self.trainR,self.testF,self.testL,self.testR,
			self.trainTimeID,self.trainStockID,validation).xgboost(mission=mission,**kwargs)
		# fit and return the mean accuracy
		if mission == 'regression':
			score = xgbTrained.score(self.testF,self.testR)
		elif mission == 'classification':
			score = xgbTrained.score(self.testF,self.testL)
		else:
			raise ValueError('The mission of xgboost should be regression or classification instead of %s'%(mission))
		return xgbTrained,score

	@funcTimer
	def lightgbm(self,mission='classification',validation=(),**kwargs):
		lgbTrained = modelSelector(self.paraTuneMethod,self.trainF,self.trainL,self.trainR,self.testF,self.testL,self.testR,
			self.trainTimeID,self.trainStockID,validation).lightgbm(mission=mission,**kwargs)
		# fit and return the mean accuracy
		if mission == 'regression':
			score = lgbTrained.score(self.testF,self.testR)
		elif mission == 'classification':
			score = lgbTrained.score(self.testF,self.testL)
		else:
			raise ValueError('The mission of light gradient boosting machine should be regression or classification instead of %s'%(mission))
		return lgbTrained,score

	@funcTimer
	def catboost(self,mission='classification',validation=(),**kwargs):
		cbTrained = modelSelector(self.paraTuneMethod,self.trainF,self.trainL,self.trainR,self.testF,self.testL,self.testR,
			self.trainTimeID,self.trainStockID,validation).catboost(mission=mission,**kwargs)
		# fit and return the mean accuracy
		if mission == 'regression':
			score = cbTrained.score(self.testF,self.testR)
		elif mission == 'classification':
			score = cbTrained.score(self.testF,self.testL)
		else:
			raise ValueError('The mission of catboost should be regression or classification instead of %s'%(mission))
		return cbTrained,score

	@funcTimer
	def logistic(self,mission='classification',validation=(),**kwargs):
		logisTrained = modelSelector(self.paraTuneMethod,self.trainF,self.trainL,self.trainR,self.testF,self.testL,self.testR,
			self.trainTimeID,self.trainStockID,validation).logistic(mission=mission,**kwargs)	
		#adaboost = super(preSetModels,self).adaboost(**kwargs)
		# fit and return the mean accuracy
		if mission != 'classification':
			raise ValueError('The mission of logistic regression should be classification instead of %s'%(mission))
		score = logisTrained.score(self.testF,self.testL)
		return logisTrained,score

	@funcTimer
	def linearSVC(self,mission='classification',validation=(),**kwargs):
		linearSVCTrained = modelSelector(self.paraTuneMethod,self.trainF,self.trainL,self.trainR,self.testF,self.testL,self.testR,
			self.trainTimeID,self.trainStockID,validation).linearSVC(mission=mission,**kwargs)	
		#adaboost = super(preSetModels,self).adaboost(**kwargs)
		# fit and return the mean accuracy
		if mission != 'classification':
			raise ValueError('The mission of linear support vector classification should be classification instead of %s'%(mission))
		score = linearSVCTrained.score(self.testF,self.testL)
		return linearSVCTrained,score

	@funcTimer
	def svm(self,mission='classification',validation=(),**kwargs):
		svmTrained = modelSelector(self.paraTuneMethod,self.trainF,self.trainL,self.trainR,self.testF,self.testL,self.testR,
			self.trainTimeID,self.trainStockID,validation).svm(mission=mission,**kwargs)	
		#adaboost = super(preSetModels,self).adaboost(**kwargs)
		# fit and return the mean accuracy
		if mission != 'classification':
			raise ValueError('The mission of support vector machine should be classification instead of %s'%(mission))
		score = svmTrained.score(self.testF,self.testL)
		return svmTrained,score

	@funcTimer
	def sgd(self,mission='classification',validation=(),**kwargs):
		sgdTrained = modelSelector(self.paraTuneMethod,self.trainF,self.trainL,self.trainR,self.testF,self.testL,self.testR,
			self.trainTimeID,self.trainStockID,validation).sgd(mission=mission,**kwargs)	
		#adaboost = super(preSetModels,self).adaboost(**kwargs)
		# fit and return the mean accuracy
		if mission == 'regression':
			score = sgdTrained.score(self.testF,self.testR)
		elif mission == 'classification':
			score = sgdTrained.score(self.testF,self.testL)
		else:
			raise ValueError('The mission of stochastic gradient descent should be regression or classification instead of %s'%(mission))
		return sgdTrained,score


	@funcTimer
	def deepforest(self,mission='classification',validation=(),**kwargs):
		dfTrained = modelSelector(self.paraTuneMethod,self.trainF,self.trainL,self.trainR,self.testF,self.testL,self.testR,
			self.trainTimeID,self.trainStockID,validation).deepforest(mission=mission,**kwargs)	
		#adaboost = super(preSetModels,self).adaboost(**kwargs)
		score = 'no score for deep forest'
		return dfTrained,score


# used for validating the models
class modelValidator(object):
	def __init__(self,model,trainX,trainY):
		self.model = model
		self.trainX = trainX
		self.trainY = trainY

	def cross_validation(self,cvNum,scoring):
		# split the data into training data and test data, then use model to fit them
		score = cross_val_score(self.model,self.trainX,self.trainY,cv=cvNum,scoring=scoring)
		print('The %s for %s times cross validation is %s'%(scoring,cvNum,score))

	def time_series_validation(self,n_splits):

		score_lst = []
		tscv = TimeSeriesSplit(n_splits=n_splits)
		trainX,trainY = self.trainX.values,self.trainY.values
		for train_index, test_index in tscv.split(trainX):
			train_X,test_X = trainX[train_index],trainX[test_index]
			train_Y,test_Y = trainY[train_index],trainY[test_index]
			model = self.model.fit(train_X,train_Y)
			score = model.score(test_X,test_Y)
			score_lst.append(score)

		print('The score for time series validation is %s'%(score_lst))


class data2Model(object):
	'''
	data encapsulation directly to models
	'''
	# TrTeDF: contains only training data and test data
	# modelMetricDF: contains all the data
	def __init__(self,TrTeDF,modelMetricDF,paraTuneMethod='NoHyperTune'):
		self.timeID = TrTeDF.timeID # time series data
		self.yieldRate = TrTeDF.yieldRate # data used for regression
		self.label = TrTeDF.label # data used for classification
		self.feature = TrTeDF.iloc[:,5:] # data served as X
		self.stockID = TrTeDF.stockID
		self.Industry = TrTeDF.citic

		self.timeID_metric = modelMetricDF.timeID
		self.stockID_metric = modelMetricDF[['timeID','stockID']]
		self.yieldRate_metric = modelMetricDF.yieldRate
		self.feature_metric = modelMetricDF.iloc[:,4:]
		
		self.paraTuneMethod = paraTuneMethod

class staticMethod(data2Model):
	'''
	static train method 
	startTrainTID: start train time id 
	endTrainTID: end train time id 
	startTrainTID/endTrainTID means the rest of timeID as test
	paraTuneMethod
	'''
	# all the data with time id between startTrainID and end TrainTID is training data, the data with time id larger than endTrainID is test data
	__trainMethod = 'staticMethod'
	#__slots__ = ()
	#paraTuneMethod = ['gridsearchcv','hyperopt']
	def __init__(self,TrTeDF,modelMetricDF,startTrainTID,endTrainTID,endTestTID,ortho_method='none',paraTuneMethod='NoHyperTune',**model_dict):
		super(staticMethod,self).__init__(TrTeDF,modelMetricDF,paraTuneMethod)
		
		self.startTrainTID = startTrainTID
		self.endTrainTID = endTrainTID
		self.endTestTID = endTestTID
		self.trainF = None 
		self.trainL = None 
		self.trainR = None  
		self.testF = None 
		self.testL = None 	#label of classification for classification train model  
		self.testR = None #label of yieldRate for regression train model  
		self.trainTimeID = None
		self.trainStockID = None
		self.filName = None 
		self.model = model_dict
		self.datapath = {}
		self.modelpath = {}
		self.ortho_method = ortho_method
		
	# split the training data and test data
	def splitTrTe(self):
		'''
		split the train and test data set 
		'''
		index1 = self.timeID[self.timeID>=self.startTrainTID].index[0]
		index2 = self.timeID[self.timeID>=self.endTrainTID].index[0]
		index3 = self.timeID[self.timeID>=self.endTestTID].index[0]
		self.startTrainTID = self.timeID[index1]
		self.endTrainTID = self.timeID[index2]
		self.endTestTID = self.timeID[index3]
		# index4 = timeIDL.index(endMonth) + 1
		index4 = np.where(self.timeID.index == index2)[0]	# index of index  
		index5 = self.timeID.index[index4][0]  	# start index for test feature
		index6 = np.where(self.timeID.index == index3)[0]
		index7 = self.timeID.index[index6][0]

		index1_ = self.timeID_metric[self.timeID_metric==self.startTrainTID].index[0]
		index2_ = self.timeID_metric[self.timeID_metric==self.endTrainTID].index[-1]
		index3_ = self.timeID_metric[self.timeID_metric==self.endTestTID].index[0]
		index4_ = np.where(self.timeID_metric.index == index2_)[0]	# index of index  
		index5_ = self.timeID_metric.index[index4_][0]  	# start index for test feature
		index6_ = np.where(self.timeID_metric.index == index3_)[0]
		index7_ = self.timeID_metric.index[index6_][0]

		# all the data with time id between startTrainID and end TrainTID is training data
		self.trainF = self.feature.loc[index1:index2]	#train features 
		self.trainL = self.label.loc[index1:index2]
		self.trainR = self.yieldRate.loc[index1:index2]
		self.trainTimeID = self.timeID.loc[index1:index2]
		self.trainStockID = self.stockID.loc[index1:index2]

		# the data with time id larger than endTrainID is test data
		self.testF = self.feature.loc[index5:index7]
		self.testL = self.label.loc[index5:index7]
		self.testR = self.yieldRate.loc[index5:index7]
		self.testMetricF = self.feature_metric[index5_:index7_]
		self.testMetricR = self.yieldRate_metric[index5_:index7_]

		self.testTimeIDStart = self.timeID[index5]
		self.filName = self.testTimeIDStart+'.m'
		
		print('Train timeID:[%s->%s].Test timeID:[%s->%s].'%(self.startTrainTID,self.endTrainTID,self.testTimeIDStart,self.endTestTID))
		# return trainF,trainL,testF,testL

	# used to set value fodel validation and model parameters
	def set_hyperopt_params(self,**kwargs):
		for key,value in kwargs.items():
			if self.model.get(key):
				self.model[key].update({'hyperopt':value})
				if not self.model[key].get('validation'):
					self.model[key].update({'validation':{}})
			else:
				self.model.update({key:{'hyperopt':value,'validation':{}}})

	def set_validation_params(self,**kwargs):
		for key,value in kwargs.items():
			if self.model.get(key):
				self.model[key].update({'validation':value})
				if not self.model[key].get('hyperopt'):
					self.model[key].update({'hyperopt':{}})
			else:
				self.model.update({key:{'hyperopt':{},'validation':value}})

	def runModels(self):
		# if the input is empty, then run all the models by default
		if not self.model:
			self.kernelRidge()
			self.ridge()
			self.lasso()
			self.elasticNet()
			self.svr()
			self.randomForest()
			self.gbdt()
			self.mlpc()
			self.gnb()
			self.adaboost()
			self.xgboost()
			self.lightgbm()
			self.catboost()
			self.svm()
			self.logistic()
			self.linearSVC()
			self.sgd()
			self.deepforest()
		else:
			if not self.model.get('model'):
				raise KeyError('key "model" should be in the model dictionary!')
			if not self.model.get('mission'):
				raise KeyError('key "mission" should be in the model dictionary!')
			if len(self.model['mission']) == 1:
				self.model.update({'mission':self.model['mission']*len(self.model['model'])})
			model_lst = self.model['model']
			mission_lst = self.model['mission']
			for i in range(len(model_lst)):
				if model_lst[i] == 'kernelridge':
					if self.model.get(model_lst[i]):
						self.kernelRidge(mission_lst[i],*[self.model[model_lst[i]]['validation']],**self.model[model_lst[i]]['hyperopt'])
					else:
						self.kernelRidge(mission=mission_lst[i])
				elif model_lst[i] == 'ridge':
					if self.model.get(model_lst[i]):
						self.ridge(mission_lst[i],*[self.model[model_lst[i]]['validation']],**self.model[model_lst[i]]['hyperopt'])
					else:
						self.ridge(mission=mission_lst[i])
				elif model_lst[i] == 'lasso':
					if self.model.get(model_lst[i]):
						self.lasso(mission_lst[i],*[self.model[model_lst[i]]['validation']],**self.model[model_lst[i]]['hyperopt'])
					else:
						self.lasso(mission=mission_lst[i])
				elif model_lst[i] == 'elasticnet':
					if self.model.get(model_lst[i]):
						self.elasticNet(mission_lst[i],*[self.model[model_lst[i]]['validation']],**self.model[model_lst[i]]['hyperopt'])
					else:
						self.elasticNet(mission=mission_lst[i])
				elif model_lst[i] == 'svr':
					if self.model.get(model_lst[i]):
						self.svr(mission_lst[i],*[self.model[model_lst[i]]['validation']],**self.model[model_lst[i]]['hyperopt'])
					else:
						self.svr(mission=mission_lst[i])
				elif model_lst[i] == 'randomforest':
					if self.model.get(model_lst[i]):
						self.randomForest(mission_lst[i],*[self.model[model_lst[i]]['validation']],**self.model[model_lst[i]]['hyperopt'])
					else:
						self.randomForest(mission=mission_lst[i])
				elif model_lst[i] == 'gbdt':
					if self.model.get(model_lst[i]):
						self.gbdt(mission_lst[i],*[self.model[model_lst[i]]['validation']],**self.model[model_lst[i]]['hyperopt'])
					else:
						self.gbdt(mission=mission_lst[i])
				elif model_lst[i] == 'mlpc':
					if self.model.get(model_lst[i]):
						self.mlpc(mission_lst[i],*[self.model[model_lst[i]]['validation']],**self.model[model_lst[i]]['hyperopt'])
					else:
						self.mlpc(mission=mission_lst[i])
				elif model_lst[i] == 'gnb':
					if self.model.get(model_lst[i]):
						self.gnb(mission_lst[i],*[self.model[model_lst[i]]['validation']],**self.model[model_lst[i]]['hyperopt'])
					else:
						self.gnb(mission=mission_lst[i])
				elif model_lst[i] == 'adaboost':
					if self.model.get(model_lst[i]):
						self.adaboost(mission_lst[i],*[self.model[model_lst[i]]['validation']],**self.model[model_lst[i]]['hyperopt'])
					else:
						self.adaboost(mission=mission_lst[i])
				elif model_lst[i] == 'xgboost':
					if self.model.get(model_lst[i]):
						self.xgboost(mission_lst[i],*[self.model[model_lst[i]]['validation']],**self.model[model_lst[i]]['hyperopt'])
					else:
						self.xgboost(mission=mission_lst[i])
				elif model_lst[i] == 'lightgbm':
					if self.model.get(model_lst[i]):
						self.lightgbm(mission_lst[i],*[self.model[model_lst[i]]['validation']],**self.model[model_lst[i]]['hyperopt'])
					else:
						self.lightgbm(mission=mission_lst[i])
				elif model_lst[i] == 'catboost':
					if self.model.get(model_lst[i]):
						self.catboost(mission_lst[i],*[self.model[model_lst[i]]['validation']],**self.model[model_lst[i]]['hyperopt'])
					else:
						self.catboost(mission=mission_lst[i])
				elif model_lst[i] == 'svm':
					if self.model.get(model_lst[i]):
						self.svm(mission_lst[i],*[self.model[model_lst[i]]['validation']],**self.model[model_lst[i]]['hyperopt'])
					else:
						self.svm(mission=mission_lst[i])
				elif model_lst[i] == 'logistic':
					if self.model.get(model_lst[i]):
						self.logistic(mission_lst[i],*[self.model[model_lst[i]]['validation']],**self.model[model_lst[i]]['hyperopt'])
					else:
						self.logistic(mission=mission_lst[i])
				elif model_lst[i] == 'linearsvc':
					if self.model.get(model_lst[i]):
						self.linearSVC(mission_lst[i],*[self.model[model_lst[i]]['validation']],**self.model[model_lst[i]]['hyperopt'])
					else:
						self.linearSVC(mission=mission_lst[i])
				elif model_lst[i] == 'sgd':
					if self.model.get(model_lst[i]):
						self.sgd(mission_lst[i],*[self.model[model_lst[i]]['validation']],**self.model[model_lst[i]]['hyperopt'])
					else:
						self.sgd(mission=mission_lst[i])
				elif model_lst[i] == 'deepforest':
					if self.model.get(model_lst[i]):
						self.deepforest(mission_lst[i],*[self.model[model_lst[i]]['validation']],**self.model[model_lst[i]]['hyperopt'])
					else:
						self.deepforest(mission=mission_lst[i])


	# train models based on user's choice
	def kernelRidge(self,mission='regression',*cvargs,**kwargs):
		# split data into training and test data
		self.splitTrTe()
		# get trained model and score
		self.kernelRidgeTrained,self.kernelRidgeTestScore = modelTrainer(self.paraTuneMethod,self.trainF,self.trainL,self.trainR,self.testF,self.testL,self.testR,
			self.trainTimeID,self.trainStockID).kernelRidge(mission=mission,validation=cvargs,**kwargs)
		print('%s_%s_%s testScore:%s'%(self.__trainMethod,'kernelridge',self.paraTuneMethod,self.kernelRidgeTestScore))
		# calculate the spearman rank IC of the trained model, here we only run once according to the definition of static method
		runningMetric_ = runningMetric(1)
		runningMetric_.modelRankIC(mission,self.testMetricF,self.testMetricR,self.kernelRidgeTrained)
		# store the trained model under the path
		fileAccess().modelStorer(self.kernelRidgeTrained,self.__trainMethod,'kernelridge',self.paraTuneMethod,self.ortho_method,self.filName)

		if len(cvargs[0]) != 0:
			param_dict = cvargs[0]

			if param_dict.get('cross_val_score'):
				cv = param_dict['cross_val_score'].get('cv',4)
				scoring = param_dict['cross_val_score'].get('cross_val_score','neg_mean_squared_error')
				modelValidator(self.kernelRidgeTrained,self.trainF,self.trainR).cross_validation(cv,scoring)

			elif param_dict.get('time_series_score'):
				n_splits = param_dict['time_series_score'].get('n_splits',5)
				modelValidator(self.kernelRidgeTrained,self.trainF,self.trainR).time_series_validation(n_splits)


	def ridge(self,mission='regression',*cvargs,**kwargs):
		self.splitTrTe()
		self.ridgeTrained,self.ridgeTestScore = modelTrainer(self.paraTuneMethod,self.trainF,self.trainL,self.trainR,self.testF,self.testL,self.testR,
			self.trainTimeID,self.trainStockID).ridge(mission=mission,validation=cvargs,**kwargs)
		print('%s_%s_%s testScore:%s'%(self.__trainMethod,'ridge',self.paraTuneMethod,self.ridgeTestScore))
		runningMetric_ = runningMetric(1)
		runningMetric_.modelRankIC(mission,self.testMetricF,self.testMetricR,self.ridgeTrained)
		fileAccess().modelStorer(self.ridgeTrained,self.__trainMethod,'ridge',self.paraTuneMethod,self.ortho_method,self.filName)

		if len(cvargs[0]) != 0:
			param_dict = cvargs[0]

			if param_dict.get('cross_val_score'):
				cv = param_dict['cross_val_score'].get('cv',4)

				if mission == 'classification':
					scoring = param_dict['cross_val_score'].get('cross_val_score','roc_auc')
					modelValidator(self.ridgeTrained,self.trainF,self.trainL).cross_validation(cv,scoring)
				elif mission == 'regression':
					scoring = param_dict['cross_val_score'].get('cross_val_score','neg_mean_squared_error')
					modelValidator(self.ridgeTrained,self.trainF,self.trainR).cross_validation(cv,scoring)
				
			elif param_dict.get('time_series_score'):
				n_splits = param_dict['time_series_score'].get('n_splits',5)

				if mission == 'classification':
					modelValidator(self.ridgeTrained,self.trainF,self.trainL).time_series_validation(n_splits)
				elif mission == 'regression':
					modelValidator(self.ridgeTrained,self.trainF,self.trainR).time_series_validation(n_splits)


	def lasso(self,mission='regression',*cvargs,**kwargs):
		self.splitTrTe()
		self.lassoTrained,self.lassoTestScore = modelTrainer(self.paraTuneMethod,self.trainF,self.trainL,self.trainR,self.testF,self.testL,self.testR,
			self.trainTimeID,self.trainStockID).lasso(mission=mission,validation=cvargs,**kwargs)
		print('%s_%s_%s testScore:%s'%(self.__trainMethod,'lasso',self.paraTuneMethod,self.lassoTestScore))
		runningMetric_ = runningMetric(1)
		runningMetric_.modelRankIC(mission,self.testMetricF,self.testMetricR,self.lassoTrained)
		fileAccess().modelStorer(self.lassoTrained,self.__trainMethod,'lasso',self.paraTuneMethod,self.ortho_method,self.filName)

		if len(cvargs[0]) != 0:
			param_dict = cvargs[0]

			if param_dict.get('cross_val_score'):
				cv = param_dict['cross_val_score'].get('cv',4)
				scoring = param_dict['cross_val_score'].get('cross_val_score','neg_mean_squared_error')
				modelValidator(self.lassoTrained,self.trainF,self.trainR).cross_validation(cv,scoring)

			elif param_dict.get('time_series_score'):
				n_splits = param_dict['time_series_score'].get('n_splits',5)
				modelValidator(self.lassoTrained,self.trainF,self.trainR).time_series_validation(n_splits)


	def elasticNet(self,mission='regression',*cvargs,**kwargs):
		self.splitTrTe()
		self.elasticNetTrained,self.elasticNetTestScore = modelTrainer(self.paraTuneMethod,self.trainF,self.trainL,self.trainR,self.testF,self.testL,self.testR,
			self.trainTimeID,self.trainStockID).elasticNet(mission=mission,validation=cvargs,**kwargs)
		print('%s_%s_%s testScore:%s'%(self.__trainMethod,'elasticnet',self.paraTuneMethod,self.elasticNetTestScore))
		runningMetric_ = runningMetric(1)
		runningMetric_.modelRankIC(mission,self.testMetricF,self.testMetricR,self.elasticNetTrained)
		fileAccess().modelStorer(self.elasticNetTrained,self.__trainMethod,'elasticnet',self.paraTuneMethod,self.ortho_method,self.filName)
		
		if len(cvargs[0]) != 0:
			param_dict = cvargs[0]

			if param_dict.get('cross_val_score'):
				cv = param_dict['cross_val_score'].get('cv',4)
				scoring = param_dict['cross_val_score'].get('cross_val_score','neg_mean_squared_error')
				modelValidator(self.elasticNetTrained,self.trainF,self.trainR).cross_validation(cv,scoring)

			elif param_dict.get('time_series_score'):
				n_splits = param_dict['time_series_score'].get('n_splits',5)
				modelValidator(self.elasticNetTrained,self.trainF,self.trainR).time_series_validation(n_splits)


	def svr(self,mission='regression',*cvargs,**kwargs):
		self.splitTrTe()
		self.svrTrained,self.svrTestScore = modelTrainer(self.paraTuneMethod,self.trainF,self.trainL,self.trainR,self.testF,self.testL,self.testR,
			self.trainTimeID,self.trainStockID).svr(mission=mission,validation=cvargs,**kwargs)
		print('%s_%s_%s testScore:%s'%(self.__trainMethod,'svr',self.paraTuneMethod,self.svrTestScore))
		runningMetric_ = runningMetric(1)
		runningMetric_.modelRankIC(mission,self.testMetricF,self.testMetricR,self.svrTrained)
		fileAccess().modelStorer(self.svrTrained,self.__trainMethod,'svr',self.paraTuneMethod,self.ortho_method,self.filName)

		if len(cvargs[0]) != 0:
			param_dict = cvargs[0]

			if param_dict.get('cross_val_score'):
				cv = param_dict['cross_val_score'].get('cv',4)
				scoring = param_dict['cross_val_score'].get('cross_val_score','neg_mean_squared_error')
				modelValidator(self.svrTrained,self.trainF,self.trainR).cross_validation(cv,scoring)

			elif param_dict.get('time_series_score'):
				n_splits = param_dict['time_series_score'].get('n_splits',5)
				modelValidator(self.svrTrained,self.trainF,self.trainR).time_series_validation(n_splits)


	#@saveLog(self.__trainMethod,self.paraTuneMethod,'trainedModel')
	def adaboost(self,mission='classification',*cvargs,**kwargs):
		'''
		*cvargs: validation in/out sample
		default :cvNum=5,testSize=0.3
		**kwargs: args dict to model  
		'''
		self.splitTrTe()
		self.adaTrained,self.adaTestScore = modelTrainer(self.paraTuneMethod,self.trainF,self.trainL,self.trainR,self.testF,self.testL,self.testR,
			self.trainTimeID,self.trainStockID).adaboost(mission=mission,validation=cvargs,**kwargs)
		print('%s_%s_%s testScore:%s'%(self.__trainMethod,'adaboost',self.paraTuneMethod,self.adaTestScore))
		#return savePath
		runningMetric_ = runningMetric(1)
		runningMetric_.modelRankIC(mission,self.testMetricF,self.testMetricR,self.adaTrained)
		fileAccess().modelStorer(self.adaTrained,self.__trainMethod,'adaboost',self.paraTuneMethod,self.ortho_method,self.filName)

		if len(cvargs[0]) != 0:
			param_dict = cvargs[0]

			if param_dict.get('cross_val_score'):
				cv = param_dict['cross_val_score'].get('cv',4)

				if mission == 'classification':
					scoring = param_dict['cross_val_score'].get('cross_val_score','roc_auc')
					modelValidator(self.adaTrained,self.trainF,self.trainL).cross_validation(cv,scoring)
				elif mission == 'regression':
					scoring = param_dict['cross_val_score'].get('cross_val_score','neg_mean_squared_error')
					modelValidator(self.adaTrained,self.trainF,self.trainR).cross_validation(cv,scoring)
				
			elif param_dict.get('time_series_score'):
				n_splits = param_dict['time_series_score'].get('n_splits',5)

				if mission == 'classification':
					modelValidator(self.adaTrained,self.trainF,self.trainL).time_series_validation(n_splits)
				elif mission == 'regression':
					modelValidator(self.adaTrained,self.trainF,self.trainR).time_series_validation(n_splits)


	def randomForest(self,mission='classification',*cvargs,**kwargs):
		self.splitTrTe()
		self.randomForestTrained,self.randomForestTestScore = modelTrainer(self.paraTuneMethod,self.trainF,self.trainL,self.trainR,self.testF,self.testL,self.testR,
			self.trainTimeID,self.trainStockID).randomForest(mission=mission,validation=cvargs,**kwargs)
		print('%s_%s_%s testScore:%s'%(self.__trainMethod,'randomforest',self.paraTuneMethod,self.randomForestTestScore))
		runningMetric_ = runningMetric(1)
		runningMetric_.modelRankIC(mission,self.testMetricF,self.testMetricR,self.randomForestTrained)
		fileAccess().modelStorer(self.randomForestTrained,self.__trainMethod,'randomforest',self.paraTuneMethod,self.ortho_method,self.filName)

		if len(cvargs[0]) != 0:
			param_dict = cvargs[0]

			if param_dict.get('cross_val_score'):
				cv = param_dict['cross_val_score'].get('cv',4)

				if mission == 'classification':
					scoring = param_dict['cross_val_score'].get('cross_val_score','roc_auc')
					modelValidator(self.randomForestTrained,self.trainF,self.trainL).cross_validation(cv,scoring)
				elif mission == 'regression':
					scoring = param_dict['cross_val_score'].get('cross_val_score','neg_mean_squared_error')
					modelValidator(self.randomForestTrained,self.trainF,self.trainR).cross_validation(cv,scoring)
				
			elif param_dict.get('time_series_score'):
				n_splits = param_dict['time_series_score'].get('n_splits',5)

				if mission == 'classification':
					modelValidator(self.randomForestTrained,self.trainF,self.trainL).time_series_validation(n_splits)
				elif mission == 'regression':
					modelValidator(self.randomForestTrained,self.trainF,self.trainR).time_series_validation(n_splits)


	def gbdt(self,mission='classification',*cvargs,**kwargs):
		self.splitTrTe()
		self.gbdtTrained,self.gbdtTestScore = modelTrainer(self.paraTuneMethod,self.trainF,self.trainL,self.trainR,self.testF,self.testL,self.testR,
			self.trainTimeID,self.trainStockID).gbdt(mission=mission,validation=cvargs,**kwargs)
		print('%s_%s_%s testScore:%s'%(self.__trainMethod,'gbdt',self.paraTuneMethod,self.gbdtTestScore))
		runningMetric_ = runningMetric(1)
		runningMetric_.modelRankIC(mission,self.testMetricF,self.testMetricR,self.gbdtTrained)
		fileAccess().modelStorer(self.gbdtTrained,self.__trainMethod,'gbdt',self.paraTuneMethod,self.ortho_method,self.filName)

		if len(cvargs[0]) != 0:
			param_dict = cvargs[0]

			if param_dict.get('cross_val_score'):
				cv = param_dict['cross_val_score'].get('cv',4)

				if mission == 'classification':
					scoring = param_dict['cross_val_score'].get('cross_val_score','roc_auc')
					modelValidator(self.gbdtTrained,self.trainF,self.trainL).cross_validation(cv,scoring)
				elif mission == 'regression':
					scoring = param_dict['cross_val_score'].get('cross_val_score','neg_mean_squared_error')
					modelValidator(self.gbdtTrained,self.trainF,self.trainR).cross_validation(cv,scoring)
				
			elif param_dict.get('time_series_score'):
				n_splits = param_dict['time_series_score'].get('n_splits',5)

				if mission == 'classification':
					modelValidator(self.gbdtTrained,self.trainF,self.trainL).time_series_validation(n_splits)
				elif mission == 'regression':
					modelValidator(self.gbdtTrained,self.trainF,self.trainR).time_series_validation(n_splits)


	def mlpc(self,mission='classification',*cvargs,**kwargs):
		self.splitTrTe()
		self.mlpcTrained,self.mlpcTestScore = modelTrainer(self.paraTuneMethod,self.trainF,self.trainL,self.trainR,self.testF,self.testL,self.testR,
			self.trainTimeID,self.trainStockID).mlpc(mission=mission,validation=cvargs,**kwargs)
		print('%s_%s_%s testScore:%s'%(self.__trainMethod,'mlpc',self.paraTuneMethod,self.mlpcTestScore))
		runningMetric_ = runningMetric(1)
		runningMetric_.modelRankIC(mission,self.testMetricF,self.testMetricR,self.mlpcTrained)
		fileAccess().modelStorer(self.mlpcTrained,self.__trainMethod,'mlpc',self.paraTuneMethod,self.ortho_method,self.filName)

		if len(cvargs[0]) != 0:
			param_dict = cvargs[0]

			if param_dict.get('cross_val_score'):
				cv = param_dict['cross_val_score'].get('cv',4)

				if mission == 'classification':
					scoring = param_dict['cross_val_score'].get('cross_val_score','roc_auc')
					modelValidator(self.mlpcTrained,self.trainF,self.trainL).cross_validation(cv,scoring)
				elif mission == 'regression':
					scoring = param_dict['cross_val_score'].get('cross_val_score','neg_mean_squared_error')
					modelValidator(self.mlpcTrained,self.trainF,self.trainR).cross_validation(cv,scoring)
				
			elif param_dict.get('time_series_score'):
				n_splits = param_dict['time_series_score'].get('n_splits',5)

				if mission == 'classification':
					modelValidator(self.mlpcTrained,self.trainF,self.trainL).time_series_validation(n_splits)
				elif mission == 'regression':
					modelValidator(self.mlpcTrained,self.trainF,self.trainR).time_series_validation(n_splits)


	def gnb(self,mission='classification',*cvargs,**kwargs):
		self.splitTrTe()
		self.gnbTrained,self.gnbTestScore = modelTrainer(self.paraTuneMethod,self.trainF,self.trainL,self.trainR,self.testF,self.testL,self.testR,
			self.trainTimeID,self.trainStockID).gnb(mission=mission,validation=cvargs,**kwargs)
		print('%s_%s_%s testScore:%s'%(self.__trainMethod,'gnb',self.paraTuneMethod,self.gnbTestScore))
		runningMetric_ = runningMetric(1)
		runningMetric_.modelRankIC(mission,self.testMetricF,self.testMetricR,self.gnbTrained)
		fileAccess().modelStorer(self.gnbTrained,self.__trainMethod,'gnb',self.paraTuneMethod,self.ortho_method,self.filName)

		if len(cvargs[0]) != 0:
			param_dict = cvargs[0]

			if param_dict.get('cross_val_score'):
				cv = param_dict['cross_val_score'].get('cv',4)
				scoring = param_dict['cross_val_score'].get('cross_val_score','roc_auc')
				modelValidator(self.gnbTrained,self.trainF,self.trainL).cross_validation(cv,scoring)

			elif param_dict.get('time_series_score'):
				n_splits = param_dict['time_series_score'].get('n_splits',5)
				modelValidator(self.gnbTrained,self.trainF,self.trainL).time_series_validation(n_splits)


	def xgboost(self,mission='classification',*cvargs,**kwargs):
		self.splitTrTe()
		self.xgbTrained,self.xgbTestScore = modelTrainer(self.paraTuneMethod,self.trainF,self.trainL,self.trainR,self.testF,self.testL,self.testR,
			self.trainTimeID,self.trainStockID).xgboost(mission=mission,validation=cvargs,**kwargs)
		print('%s_%s_%s testScore:%s'%(self.__trainMethod,'xgboost',self.paraTuneMethod,self.xgbTestScore))
		runningMetric_ = runningMetric(1)
		runningMetric_.modelRankIC(mission,self.testMetricF,self.testMetricR,self.xgbTrained)
		fileAccess().modelStorer(self.xgbTrained,self.__trainMethod,'xgboost',self.paraTuneMethod,self.ortho_method,self.filName)

		if len(cvargs[0]) != 0:
			param_dict = cvargs[0]

			if param_dict.get('cross_val_score'):
				cv = param_dict['cross_val_score'].get('cv',4)

				if mission == 'classification':
					scoring = param_dict['cross_val_score'].get('cross_val_score','roc_auc')
					modelValidator(self.xgbTrained,self.trainF,self.trainL).cross_validation(cv,scoring)
				elif mission == 'regression':
					scoring = param_dict['cross_val_score'].get('cross_val_score','neg_mean_squared_error')
					modelValidator(self.xgbTrained,self.trainF,self.trainR).cross_validation(cv,scoring)
				
			elif param_dict.get('time_series_score'):
				n_splits = param_dict['time_series_score'].get('n_splits',5)

				if mission == 'classification':
					modelValidator(self.xgbTrained,self.trainF,self.trainL).time_series_validation(n_splits)
				elif mission == 'regression':
					modelValidator(self.xgbTrained,self.trainF,self.trainR).time_series_validation(n_splits)


	#@saveLog(self.__trainMethod,self.paraTuneMethod,'trainedModel')
	def lightgbm(self,mission='classification',*cvargs,**kwargs):
		self.splitTrTe()
		self.lgbTrained,self.lgbTestScore = modelTrainer(self.paraTuneMethod,self.trainF,self.trainL,self.trainR,self.testF,self.testL,self.testR,
			self.trainTimeID,self.trainStockID).lightgbm(mission=mission,validation=cvargs,**kwargs)
		
		print('%s_%s_%s testScore:%s'%(self.__trainMethod,'lightgbm',self.paraTuneMethod,self.lgbTestScore))
		runningMetric_ = runningMetric(1)
		runningMetric_.modelRankIC(mission,self.testMetricF,self.testMetricR,self.lgbTrained)
		fileAccess().modelStorer(self.lgbTrained,self.__trainMethod,'lightgbm',self.paraTuneMethod,self.ortho_method,self.filName)

		if len(cvargs[0]) != 0:
			param_dict = cvargs[0]
			if param_dict.get('lightgbm'):
				cv, scoring = at(param_dict['lightgbm'],'cv','scoring')

		if len(cvargs[0]) != 0:
			param_dict = cvargs[0]

			if param_dict.get('cross_val_score'):
				cv = param_dict['cross_val_score'].get('cv',4)

				if mission == 'classification':
					scoring = param_dict['cross_val_score'].get('cross_val_score','roc_auc')
					modelValidator(self.lgbTrained,self.trainF,self.trainL).cross_validation(cv,scoring)
				elif mission == 'regression':
					scoring = param_dict['cross_val_score'].get('cross_val_score','neg_mean_squared_error')
					modelValidator(self.lgbTrained,self.trainF,self.trainR).cross_validation(cv,scoring)
				
			elif param_dict.get('time_series_score'):
				n_splits = param_dict['time_series_score'].get('n_splits',5)

				if mission == 'classification':
					modelValidator(self.lgbTrained,self.trainF,self.trainL).time_series_validation(n_splits)
				elif mission == 'regression':
					modelValidator(self.lgbTrained,self.trainF,self.trainR).time_series_validation(n_splits)


	#@saveLog(self.__trainMethod,self.paraTuneMethod,'trainedModel')
	def catboost(self,mission='classification',*cvargs,**kwargs):
		self.splitTrTe()
		self.cbTrained,self.cbTestScore = modelTrainer(self.paraTuneMethod,self.trainF,self.trainL,self.trainR,self.testF,self.testL,self.testR,
			self.trainTimeID,self.trainStockID).catboost(mission=mission,validation=cvargs,**kwargs)
		print('%s_%s_%s testScore:%s'%(self.__trainMethod,'catboost',self.paraTuneMethod,self.cbTestScore))
		runningMetric_ = runningMetric(1)
		runningMetric_.modelRankIC(mission,self.testMetricF,self.testMetricR,self.cbTrained)
		fileAccess().modelStorer(self.cbTrained,self.__trainMethod,'catboost',self.paraTuneMethod,self.ortho_method,self.filName)

		if len(cvargs[0]) != 0:
			param_dict = cvargs[0]

			if param_dict.get('cross_val_score'):
				cv = param_dict['cross_val_score'].get('cv',4)

				if mission == 'classification':
					scoring = param_dict['cross_val_score'].get('cross_val_score','roc_auc')
					modelValidator(self.cbTrained,self.trainF,self.trainL).cross_validation(cv,scoring)
				elif mission == 'regression':
					scoring = param_dict['cross_val_score'].get('cross_val_score','neg_mean_squared_error')
					modelValidator(self.cbTrained,self.trainF,self.trainR).cross_validation(cv,scoring)
				
			elif param_dict.get('time_series_score'):
				n_splits = param_dict['time_series_score'].get('n_splits',5)

				if mission == 'classification':
					modelValidator(self.cbTrained,self.trainF,self.trainL).time_series_validation(n_splits)
				elif mission == 'regression':
					modelValidator(self.cbTrained,self.trainF,self.trainR).time_series_validation(n_splits)


	def logistic(self,mission='classification',*cvargs,**kwargs):
		self.splitTrTe()
		self.logisticTrained,self.logisticTestScore = modelTrainer(self.paraTuneMethod,self.trainF,self.trainL,self.trainR,self.testF,self.testL,self.testR,
			self.trainTimeID,self.trainStockID).logistic(mission=mission,validation=cvargs,**kwargs)
		print('%s_%s_%s testScore:%s'%(self.__trainMethod,'logistic',self.paraTuneMethod,self.logisticTestScore))
		runningMetric_ = runningMetric(1)
		runningMetric_.modelRankIC(mission,self.testMetricF,self.testMetricR,self.logisticTrained)
		fileAccess().modelStorer(self.logisticTrained,self.__trainMethod,'logistic',self.paraTuneMethod,self.ortho_method,self.filName)

		if len(cvargs[0]) != 0:
			param_dict = cvargs[0]

			if param_dict.get('cross_val_score'):
				scoring = param_dict['cross_val_score'].get('cross_val_score','roc_auc')
				cv = param_dict['cross_val_score'].get('cv',4)
				modelValidator(self.logisticTrained,self.trainF,self.trainL).cross_validation(cv,scoring)

			elif param_dict.get('time_series_score'):
				n_splits = param_dict['time_series_score'].get('n_splits',5)
				modelValidator(self.logisticTrained,self.trainF,self.trainL).time_series_validation(n_splits)


	def sgd(self,mission='classification',*cvargs,**kwargs):
		self.splitTrTe()
		self.sgdTrained,self.sgdTestScore = modelTrainer(self.paraTuneMethod,self.trainF,self.trainL,self.trainR,self.testF,self.testL,self.testR,
			self.trainTimeID,self.trainStockID).sgd(mission=mission,validation=cvargs,**kwargs)
		print('%s_%s_%s testScore:%s'%(self.__trainMethod,'sgd',self.paraTuneMethod,self.sgdTestScore))
		runningMetric_ = runningMetric(1)
		runningMetric_.modelRankIC(mission,self.testMetricF,self.testMetricR,self.sgdTrained)
		fileAccess().modelStorer(self.sgdTrained,self.__trainMethod,'sgd',self.paraTuneMethod,self.ortho_method,self.filName)

		if len(cvargs[0]) != 0:
			param_dict = cvargs[0]

			if param_dict.get('cross_val_score'):
				cv = param_dict['cross_val_score'].get('cv',4)

				if mission == 'classification':
					scoring = param_dict['cross_val_score'].get('cross_val_score','roc_auc')
					modelValidator(self.sgdTrained,self.trainF,self.trainL).cross_validation(cv,scoring)
				elif mission == 'regression':
					scoring = param_dict['cross_val_score'].get('cross_val_score','neg_mean_squared_error')
					modelValidator(self.sgdTrained,self.trainF,self.trainR).cross_validation(cv,scoring)
				
			elif param_dict.get('time_series_score'):
				n_splits = param_dict['time_series_score'].get('n_splits',5)

				if mission == 'classification':
					modelValidator(self.sgdTrained,self.trainF,self.trainL).time_series_validation(n_splits)
				elif mission == 'regression':
					modelValidator(self.sgdTrained,self.trainF,self.trainR).time_series_validation(n_splits)


	def svm(self,mission='classification',*cvargs,**kwargs):
		self.splitTrTe()
		self.svmTrained,self.svmTestScore = modelTrainer(self.paraTuneMethod,self.trainF,self.trainL,self.trainR,self.testF,self.testL,self.testR,
			self.trainTimeID,self.trainStockID).svm(mission=mission,validation=cvargs,**kwargs)
		print('%s_%s_%s testScore:%s'%(self.__trainMethod,'svm',self.paraTuneMethod,self.svmTestScore))
		runningMetric_ = runningMetric(1)
		runningMetric_.modelRankIC(mission,self.testMetricF,self.testMetricR,self.svmTrained)
		fileAccess().modelStorer(self.svmTrained,self.__trainMethod,'svm',self.paraTuneMethod,self.ortho_method,self.filName)

		if len(cvargs[0]) != 0:
			param_dict = cvargs[0]

			if param_dict.get('cross_val_score'):
				cv = param_dict['cross_val_score'].get('cv',4)
				scoring = param_dict['cross_val_score'].get('cross_val_score','roc_auc')
				modelValidator(self.svmTrained,self.trainF,self.trainL).cross_validation(cv,scoring)

			elif param_dict.get('time_series_score'):
				n_splits = param_dict['time_series_score'].get('n_splits',5)
				modelValidator(self.svmTrained,self.trainF,self.trainL).time_series_validation(n_splits)


	def linearSVC(self,mission='classification',*cvargs,**kwargs):
		self.splitTrTe()
		self.linearSVCTrained,self.linearSVCTestScore = modelTrainer(self.paraTuneMethod,self.trainF,self.trainL,self.trainR,self.testF,self.testL,self.testR,
			self.trainTimeID,self.trainStockID).linearSVC(mission=mission,validation=cvargs,**kwargs)
		print('%s_%s_%s testScore:%s'%(self.__trainMethod,'linearsvc',self.paraTuneMethod,self.linearSVCTestScore))
		runningMetric_ = runningMetric(1)
		runningMetric_.modelRankIC(mission,self.testMetricF,self.testMetricR,self.linearSVCTrained)
		fileAccess().modelStorer(self.linearSVCTrained,self.__trainMethod,'linearsvc',self.paraTuneMethod,self.ortho_method,self.filName)

		if len(cvargs[0]) != 0:
			param_dict = cvargs[0]

			if param_dict.get('cross_val_score'):
				cv = param_dict['cross_val_score'].get('cv',4)
				scoring = param_dict['cross_val_score'].get('cross_val_score','roc_auc')
				modelValidator(self.linearSVCTrained,self.trainF,self.trainL).cross_validation(cv,scoring)

			elif param_dict.get('time_series_score'):
				n_splits = param_dict['time_series_score'].get('n_splits',5)
				modelValidator(self.linearSVCTrained,self.trainF,self.trainL).time_series_validation(n_splits)


	def deepforest(self,mission='classification',*cvargs,**kwargs):
		self.splitTrTe()
		self.dfTrained,self.dfTestScore = modelTrainer(self.paraTuneMethod,self.trainF,self.trainL,self.trainR,self.testF,self.testL,self.testR,
			self.trainTimeID,self.trainStockID).deepforest(mission=mission,validation=cvargs,**kwargs)
		print('%s_%s_%s testScore:%s'%(self.__trainMethod,'deepforest',self.paraTuneMethod,self.dfTestScore))
		runningMetric_ = runningMetric(1)
		runningMetric_.modelRankIC(mission,self.testMetricF,self.testMetricR,self.dfTrained)
		fileAccess().modelStorer(self.dfTrained,self.__trainMethod,'deepforest',self.paraTuneMethod,self.ortho_method,self.filName)

		if len(cvargs[0]) != 0:
			param_dict = cvargs[0]

			if param_dict.get('cross_val_score'):
				cv = param_dict['cross_val_score'].get('cv',4)

				if mission == 'classification':
					scoring = param_dict['cross_val_score'].get('cross_val_score','roc_auc')
					modelValidator(self.dfTrained,self.trainF,self.trainL).cross_validation(cv,scoring)
				elif mission == 'regression':
					scoring = param_dict['cross_val_score'].get('cross_val_score','neg_mean_squared_error')
					modelValidator(self.dfTrained,self.trainF,self.trainR).cross_validation(cv,scoring)
				
			elif param_dict.get('time_series_score'):
				n_splits = param_dict['time_series_score'].get('n_splits',5)

				if mission == 'classification':
					modelValidator(self.dfTrained,self.trainF,self.trainL).time_series_validation(n_splits)
				elif mission == 'regression':
					modelValidator(self.dfTrained,self.trainF,self.trainR).time_series_validation(n_splits)


# trailing method
class trailingMethod(data2Model):
	'''
	trailing train with trainWindowL timeIDs as train window
	startTrainTID: the first trailing train timeID. So the first train window will be [startTrainTID,startTrainTID+trainWindowL-1]
	'''

	__trainMethod = 'trailingMethod'
	__modelNames = ('kernelRidge','ridge','lasso','elasticNet','svr','randomForest','gbdt','mlpc','gnb','adaboost','xgboost','lightgbm','catboost','svm','logistic','linearSVC','sgd','deepforest')
	 
	#paraTuneMethod = ['gridsearchcv','hyperopt']
	def __init__(self,TrTeDF,modelMetricDF,startTrainTID,ortho_method='none',trainWindowL=12,window=1,paraTuneMethod='NoHyperTune',**model_dict):
		super(trailingMethod,self).__init__(TrTeDF,modelMetricDF,paraTuneMethod)
		
		self.startTrainTID = startTrainTID	
		self.trainWindowL = trainWindowL
		self.window = window
		self.indexRML = None
		self.testIDNum = None 
		self.model = model_dict
		self.ortho_method = ortho_method

	# get the index of the initial start date and end date
	def timeIndexInitial(self):

		timeIDUQL = pd.unique(self.timeID).tolist()	
		# get the index of time id of the start date and end date
		try:
			indexSTID = timeIDUQL.index(self.startTrainTID)
		except:
			temp_lst = sorted(timeIDUQL+[self.startTrainTID])
			indexSTID = temp_lst.index(self.startTrainTID)
			self.startTrainTID = timeIDUQL[indexSTID]
		indexETID = indexSTID + (self.trainWindowL-1)
		# get the index of time id of dates after end date
		self.indexRML = list(range(indexETID+1,len(timeIDUQL),self.window))
		self.testIDNum = len(self.indexRML)	
		print('%s timeIDs to test'%self.testIDNum)

		return timeIDUQL,indexSTID,indexETID

	# return generator used to generate training data and test data for each period
	def TrTeGenerator(self,timeIDUQL,indexSTID,indexETID):
		'''
		generate train ans test data set follow the trailing windows
		'''

		trainIDS = self.startTrainTID		#the first start train timeID 
		trainIDE = timeIDUQL[indexETID]	#the first end train timeID 

		for i in range(0,len(self.indexRML)):
			# get training data and test data for a specific period
			testTimeID_i = timeIDUQL[self.indexRML[i]]
			testF_i = self.feature[self.timeID==testTimeID_i]
			testL_i = self.label[self.timeID==testTimeID_i]
			testR_i = self.yieldRate[self.timeID==testTimeID_i]
			testMetricF_i = self.feature_metric[self.timeID_metric==testTimeID_i]
			testMetricR_i = self.yieldRate_metric[self.timeID_metric==testTimeID_i]
			testStockID_i = self.stockID_metric[self.timeID_metric==testTimeID_i]

			print('Test timeID:%s'%testTimeID_i)

			index1 = self.timeID[self.timeID==trainIDS].index[0]
			index2 = self.timeID[self.timeID==trainIDE].index[-1]
			print('Train timeID:[%s->%s]'%(trainIDS,trainIDE))
			
			trainF_i = self.feature.loc[index1:index2]
			trainL_i = self.label.loc[index1:index2]
			trainR_i = self.yieldRate.loc[index1:index2]
			trainTimeID_i = self.timeID.loc[index1:index2]
			trainStockID_i = self.stockID.loc[index1:index2]

			modelfilName_i = testTimeID_i+'.m'
			datafilName_i = testTimeID_i+'.pkl'

			yield(trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i,modelfilName_i,
				datafilName_i,testTimeID_i,testMetricF_i,testMetricR_i,testStockID_i)

			indexSTID += self.window
			indexETID += self.window
			trainIDS = timeIDUQL[indexSTID]
			trainIDE = timeIDUQL[indexETID]

	def set_hyperopt_params(self,**kwargs):
		for key,value in kwargs.items():
			if self.model.get(key):
				self.model[key].update({'hyperopt':value})
				if not self.model[key].get('validation'):
					self.model[key].update({'validation':{}})
			else:
				self.model.update({key:{'hyperopt':value,'validation':{}}})

	def set_validation_params(self,**kwargs):
		for key,value in kwargs.items():
			if self.model.get(key):
				self.model[key].update({'validation':value})
				if not self.model[key].get('hyperopt'):
					self.model[key].update({'hyperopt':{}})
			else:
				self.model.update({key:{'hyperopt':{},'validation':value}})

	def runModels(self):
		
		if not self.model:
			self.kernelRidge()
			self.ridge()
			self.lasso()
			self.elasticNet()
			self.svr()
			self.randomForest()
			self.gbdt()
			self.mlpc()
			self.gnb()
			self.adaboost()
			self.xgboost()
			self.lightgbm()
			self.catboost()
			self.svm()
			self.logistic()
			self.linearSVC()
			self.sgd()
			self.deepforest()
		else:
			if not self.model.get('model'):
				raise KeyError('key "model" should be in the model dictionary!')
			if not self.model.get('mission'):
				raise KeyError('key "mission" should be in the model dictionary!')
			if len(self.model['mission']) == 1:
				self.model.update({'mission':self.model['mission']*len(self.model['model'])})
			model_lst = self.model['model']
			mission_lst = self.model['mission']
			for i in range(len(model_lst)):
				if model_lst[i] == 'kernelridge':
					if self.model.get(model_lst[i]):
						self.kernelRidge(mission_lst[i],*[self.model[model_lst[i]]['validation']],**self.model[model_lst[i]]['hyperopt'])
					else:
						self.kernelRidge(mission=mission_lst[i])
				elif model_lst[i] == 'ridge':
					if self.model.get(model_lst[i]):
						self.ridge(mission_lst[i],*[self.model[model_lst[i]]['validation']],**self.model[model_lst[i]]['hyperopt'])
					else:
						self.ridge(mission=mission_lst[i])
				elif model_lst[i] == 'lasso':
					if self.model.get(model_lst[i]):
						self.lasso(mission_lst[i],*[self.model[model_lst[i]]['validation']],**self.model[model_lst[i]]['hyperopt'])
					else:
						self.lasso(mission=mission_lst[i])
				elif model_lst[i] == 'elasticnet':
					if self.model.get(model_lst[i]):
						self.elasticNet(mission_lst[i],*[self.model[model_lst[i]]['validation']],**self.model[model_lst[i]]['hyperopt'])
					else:
						self.elasticNet(mission=mission_lst[i])
				elif model_lst[i] == 'svr':
					if self.model.get(model_lst[i]):
						self.svr(mission_lst[i],*[self.model[model_lst[i]]['validation']],**self.model[model_lst[i]]['hyperopt'])
					else:
						self.svr(mission=mission_lst[i])
				elif model_lst[i] == 'randomforest':
					if self.model.get(model_lst[i]):
						self.randomForest(mission_lst[i],*[self.model[model_lst[i]]['validation']],**self.model[model_lst[i]]['hyperopt'])
					else:
						self.randomForest(mission=mission_lst[i])
				elif model_lst[i] == 'gbdt':
					if self.model.get(model_lst[i]):
						self.gbdt(mission_lst[i],*[self.model[model_lst[i]]['validation']],**self.model[model_lst[i]]['hyperopt'])
					else:
						self.gbdt(mission=mission_lst[i])
				elif model_lst[i] == 'mlpc':
					if self.model.get(model_lst[i]):
						self.mlpc(mission_lst[i],*[self.model[model_lst[i]]['validation']],**self.model[model_lst[i]]['hyperopt'])
					else:
						self.mlpc(mission=mission_lst[i])
				elif model_lst[i] == 'gnb':
					if self.model.get(model_lst[i]):
						self.gnb(mission_lst[i],*[self.model[model_lst[i]]['validation']],**self.model[model_lst[i]]['hyperopt'])
					else:
						self.gnb(mission=mission_lst[i])
				elif model_lst[i] == 'adaboost':
					if self.model.get(model_lst[i]):
						self.adaboost(mission_lst[i],*[self.model[model_lst[i]]['validation']],**self.model[model_lst[i]]['hyperopt'])
					else:
						self.adaboost(mission=mission_lst[i])
				elif model_lst[i] == 'xgboost':
					if self.model.get(model_lst[i]):
						self.xgboost(mission_lst[i],*[self.model[model_lst[i]]['validation']],**self.model[model_lst[i]]['hyperopt'])
					else:
						self.xgboost(mission=mission_lst[i])
				elif model_lst[i] == 'lightgbm':
					if self.model.get(model_lst[i]):
						self.lightgbm(mission_lst[i],*[self.model[model_lst[i]]['validation']],**self.model[model_lst[i]]['hyperopt'])
					else:
						self.lightgbm(mission=mission_lst[i])
				elif model_lst[i] == 'catboost':
					if self.model.get(model_lst[i]):
						self.catboost(mission_lst[i],*[self.model[model_lst[i]]['validation']],**self.model[model_lst[i]]['hyperopt'])
					else:
						self.catboost(mission=mission_lst[i])
				elif model_lst[i] == 'svm':
					if self.model.get(model_lst[i]):
						self.svm(mission_lst[i],*[self.model[model_lst[i]]['validation']],**self.model[model_lst[i]]['hyperopt'])
					else:
						self.svm(mission=mission_lst[i])
				elif model_lst[i] == 'logistic':
					if self.model.get(model_lst[i]):
						self.logistic(mission_lst[i],*[self.model[model_lst[i]]['validation']],**self.model[model_lst[i]]['hyperopt'])
					else:
						self.logistic(mission=mission_lst[i])
				elif model_lst[i] == 'linearsvc':
					if self.model.get(model_lst[i]):
						self.linearSVC(mission_lst[i],*[self.model[model_lst[i]]['validation']],**self.model[model_lst[i]]['hyperopt'])
					else:
						self.linearSVC(mission=mission_lst[i])
				elif model_lst[i] == 'sgd':
					if self.model.get(model_lst[i]):
						self.sgd(mission_lst[i],*[self.model[model_lst[i]]['validation']],**self.model[model_lst[i]]['hyperopt'])
					else:
						self.sgd(mission=mission_lst[i])
				elif model_lst[i] == 'deepforest':
					if self.model.get(model_lst[i]):
						self.deepforest(mission_lst[i],*[self.model[model_lst[i]]['validation']],**self.model[model_lst[i]]['hyperopt'])
					else:
						self.deepforest(mission=mission_lst[i])


	@timeLog('trailingMethod','total')
	def kernelRidge(self,mission='regression',*cvargs,**kwargs):
		# get the initial index for time id
		timeIDUQL,indexSTID,indexETID = self.timeIndexInitial()
		# get the generator used to generate training and test data
		generator = self.TrTeGenerator(timeIDUQL,indexSTID,indexETID)
		runningMetric1 = runningMetric(self.testIDNum)
		# the value of testIDNum depends on the number of remaining date
		while self.testIDNum > 0: 
			trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i,modelfilName_i,datafilName_i,testTimeID_i,\
				testMetricF_i,testMetricR_i,testStockID_i = next(generator)		#generator.next() for python 2.x
			self.kernelRidgeTrained_i,self.kernelRidgeTestScore_i = \
				modelTrainer(self.paraTuneMethod,trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i)\
					.kernelRidge(mission=mission,validation=cvargs,**kwargs)
			#rankIC_i = runningMetric().modelRankIC(kernelRidgeTrained_i,testTimeID_i)
			print('%s_%s_%s testScore:%s'%(self.__trainMethod,'kernelridge',self.paraTuneMethod,self.kernelRidgeTestScore_i))
			runningMetric1.modelRankIC(mission,testMetricF_i,testMetricR_i,self.kernelRidgeTrained_i)
			fileAccess().modelStorer(self.kernelRidgeTrained_i,self.__trainMethod,'kernelridge',self.paraTuneMethod,self.ortho_method,modelfilName_i)


			if len(cvargs[0]) != 0:
				param_dict = cvargs[0]

				if param_dict.get('cross_val_score'):
					cv = param_dict['cross_val_score'].get('cv',4)
					scoring = param_dict['cross_val_score'].get('cross_val_score','neg_mean_squared_error')
					modelValidator(self.kernelRidgeTrained_i,trainF_i,trainR_i).cross_validation(cv,scoring)

				elif param_dict.get('time_series_score'):
					n_splits = param_dict['time_series_score'].get('n_splits',5)
					modelValidator(self.kernelRidgeTrained_i,trainF_i,trainR_i).time_series_validation(n_splits)

			self.testIDNum -= 1
			

	@timeLog('trailingMethod','total')
	def ridge(self,mission='regression',*cvargs,**kwargs):
		timeIDUQL,indexSTID,indexETID = self.timeIndexInitial()
		generator = self.TrTeGenerator(timeIDUQL,indexSTID,indexETID)
		runningMetric1 = runningMetric(self.testIDNum)
		while self.testIDNum > 0: 
			trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i,modelfilName_i,datafilName_i,testTimeID_i,\
				testMetricF_i,testMetricR_i,testStockID_i = next(generator)		#generator.next() for python 2.x
			self.ridgeTrained_i,self.ridgeTestScore_i = \
				modelTrainer(self.paraTuneMethod,trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i)\
					.ridge(mission=mission,validation=cvargs,**kwargs)
			print('%s_%s_%s testScore:%s'%(self.__trainMethod,'ridge',self.paraTuneMethod,self.ridgeTestScore_i))
			runningMetric1.modelRankIC(mission,testMetricF_i,testMetricR_i,self.ridgeTrained_i)
			fileAccess().modelStorer(self.ridgeTrained_i,self.__trainMethod,'ridge',self.paraTuneMethod,self.ortho_method,modelfilName_i)


			if len(cvargs[0]) != 0:
				param_dict = cvargs[0]

				if param_dict.get('cross_val_score'):
					cv = param_dict['cross_val_score'].get('cv',4)

					if mission == 'classification':
						scoring = param_dict['cross_val_score'].get('cross_val_score','roc_auc')
						modelValidator(self.kernelRidgeTrained_i,trainF_i,trainL_i).cross_validation(cv,scoring)
					elif mission == 'regression':
						scoring = param_dict['cross_val_score'].get('cross_val_score','neg_mean_squared_error')
						modelValidator(self.ridgeTrained_i,trainF_i,trainR_i).cross_validation(cv,scoring)
					
				elif param_dict.get('time_series_score'):
					n_splits = param_dict['time_series_score'].get('n_splits',5)

					if mission == 'classification':
						modelValidator(self.ridgeTrained_i,trainF_i,trainL_i).time_series_validation(n_splits)
					elif mission == 'regression':
						modelValidator(self.ridgeTrained_i,trainF_i,trainR_i).time_series_validation(n_splits)

			self.testIDNum -= 1
			

	@timeLog('trailingMethod','total')
	def lasso(self,mission='regression',*cvargs,**kwargs):
		timeIDUQL,indexSTID,indexETID = self.timeIndexInitial()
		generator = self.TrTeGenerator(timeIDUQL,indexSTID,indexETID)
		runningMetric1 = runningMetric(self.testIDNum)
		while self.testIDNum > 0: 
			trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i,modelfilName_i,datafilName_i,testTimeID_i,\
				testMetricF_i,testMetricR_i,testStockID_i = next(generator)		#generator.next() for python 2.x
			self.lassoTrained_i,self.lassoTestScore_i = \
				modelTrainer(self.paraTuneMethod,trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i)\
					.lasso(mission=mission,validation=cvargs,**kwargs)
			print('%s_%s_%s testScore:%s'%(self.__trainMethod,'lasso',self.paraTuneMethod,self.lassoTestScore_i))
			runningMetric1.modelRankIC(mission,testMetricF_i,testMetricR_i,self.lassoTrained_i)
			fileAccess().modelStorer(self.lassoTrained_i,self.__trainMethod,'lasso',self.paraTuneMethod,self.ortho_method,modelfilName_i)


			if len(cvargs[0]) != 0:
				param_dict = cvargs[0]

				if param_dict.get('cross_val_score'):
					cv = param_dict['cross_val_score'].get('cv',4)
					scoring = param_dict['cross_val_score'].get('cross_val_score','neg_mean_squared_error')
					modelValidator(self.lassoTrained_i,trainF_i,trainR_i).cross_validation(cv,scoring)

				elif param_dict.get('time_series_score'):
					n_splits = param_dict['time_series_score'].get('n_splits',5)
					modelValidator(self.lassoTrained_i,trainF_i,trainR_i).time_series_validation(n_splits)

			self.testIDNum -= 1
			

	@timeLog('trailingMethod','total')
	def elasticNet(self,mission='regression',*cvargs,**kwargs):
		timeIDUQL,indexSTID,indexETID = self.timeIndexInitial()
		generator = self.TrTeGenerator(timeIDUQL,indexSTID,indexETID)
		runningMetric1 = runningMetric(self.testIDNum)
		while self.testIDNum > 0: 
			trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i,modelfilName_i,datafilName_i,testTimeID_i,\
				testMetricF_i,testMetricR_i,testStockID_i = next(generator)		#generator.next() for python 2.x
			self.elasticNetTrained_i,self.elasticNetTestScore_i = \
				modelTrainer(self.paraTuneMethod,trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i)\
					.elasticNet(mission=mission,validation=cvargs,**kwargs)
			print('%s_%s_%s testScore:%s'%(self.__trainMethod,'elasticnet',self.paraTuneMethod,self.elasticNetTestScore_i))
			runningMetric1.modelRankIC(mission,testMetricF_i,testMetricR_i,self.elasticNetTrained_i)
			fileAccess().modelStorer(self.elasticNetTrained_i,self.__trainMethod,'elasticnet',self.paraTuneMethod,self.ortho_method,modelfilName_i)


			if len(cvargs[0]) != 0:
				param_dict = cvargs[0]

				if param_dict.get('cross_val_score'):
					cv = param_dict['cross_val_score'].get('cv',4)
					scoring = param_dict['cross_val_score'].get('cross_val_score','neg_mean_squared_error')
					modelValidator(self.elasticNetTrained_i,trainF_i,trainR_i).cross_validation(cv,scoring)

				elif param_dict.get('time_series_score'):
					n_splits = param_dict['time_series_score'].get('n_splits',5)
					modelValidator(self.elasticNetTrained_i,trainF_i,trainR_i).time_series_validation(n_splits)

			self.testIDNum -= 1
			

	@timeLog('trailingMethod','total')
	def svr(self,mission='regression',*cvargs,**kwargs):
		timeIDUQL,indexSTID,indexETID = self.timeIndexInitial()
		generator = self.TrTeGenerator(timeIDUQL,indexSTID,indexETID)
		runningMetric1 = runningMetric(self.testIDNum)
		while self.testIDNum > 0: 
			trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i,modelfilName_i,datafilName_i,testTimeID_i,\
				testMetricF_i,testMetricR_i,testStockID_i = next(generator)		#generator.next() for python 2.x
			self.svrTrained_i,self.svrTestScore_i = \
				modelTrainer(self.paraTuneMethod,trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i)\
					.svr(mission=mission,validation=cvargs,**kwargs)
			print('%s_%s_%s testScore:%s'%(self.__trainMethod,'svr',self.paraTuneMethod,self.svrTestScore_i))
			runningMetric1.modelRankIC(mission,testMetricF_i,testMetricR_i,self.svrTrained_i)
			fileAccess().modelStorer(self.svrTrained_i,self.__trainMethod,'svr',self.paraTuneMethod,self.ortho_method,modelfilName_i)


			if len(cvargs[0]) != 0:
				param_dict = cvargs[0]

				if param_dict.get('cross_val_score'):
					cv = param_dict['cross_val_score'].get('cv',4)
					scoring = param_dict['cross_val_score'].get('cross_val_score','neg_mean_squared_error')
					modelValidator(self.svrTrained_i,trainF_i,trainR_i).cross_validation(cv,scoring)

				elif param_dict.get('time_series_score'):
					n_splits = param_dict['time_series_score'].get('n_splits',5)
					modelValidator(self.svrTrained_i,self.trainF,self.trainR).time_series_validation(n_splits)

			self.testIDNum -= 1


	@timeLog('trailingMethod','total')
	def randomForest(self,mission='classification',*cvargs,**kwargs):
		timeIDUQL,indexSTID,indexETID = self.timeIndexInitial()
		generator = self.TrTeGenerator(timeIDUQL,indexSTID,indexETID)
		runningMetric1 = runningMetric(self.testIDNum)
		while self.testIDNum > 0: 
			trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i,modelfilName_i,datafilName_i,testTimeID_i,\
				testMetricF_i,testMetricR_i,testStockID_i = next(generator)		#generator.next() for python 2.x
			self.randomForestTrained_i,self.randomForestTestScore_i = \
				modelTrainer(self.paraTuneMethod,trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i)\
					.randomForest(mission=mission,validation=cvargs,**kwargs)
			print('%s_%s_%s testScore:%s'%(self.__trainMethod,'randomforest',self.paraTuneMethod,self.randomForestTestScore_i))
			runningMetric1.modelRankIC(mission,testMetricF_i,testMetricR_i,self.randomForestTrained_i)
			fileAccess().modelStorer(self.randomForestTrained_i,self.__trainMethod,'randomforest',self.paraTuneMethod,self.ortho_method,modelfilName_i)


			if len(cvargs[0]) != 0:
				param_dict = cvargs[0]

				if param_dict.get('cross_val_score'):
					cv = param_dict['cross_val_score'].get('cv',4)


					if mission == 'classification':
						scoring = param_dict['cross_val_score'].get('cross_val_score','roc_auc')
						modelValidator(self.randomForestTrained_i,self.trainF,self.trainL).cross_validation(cv,scoring)
					elif mission == 'regression':
						scoring = param_dict['cross_val_score'].get('cross_val_score','neg_mean_squared_error')
						modelValidator(self.randomForestTrained_i,self.trainF,self.trainR).cross_validation(cv,scoring)
					
				elif param_dict.get('time_series_score'):
					n_splits = param_dict['time_series_score'].get('n_splits',5)

					if mission == 'classification':
						modelValidator(self.randomForestTrained_i,self.trainF,self.trainL).time_series_validation(n_splits)
					elif mission == 'regression':
						modelValidator(self.randomForestTrained_i,self.trainF,self.trainR).time_series_validation(n_splits)

			self.testIDNum -= 1
			

	@timeLog('trailingMethod','total')
	def gbdt(self,mission='classification',*cvargs,**kwargs):
		timeIDUQL,indexSTID,indexETID = self.timeIndexInitial()
		generator = self.TrTeGenerator(timeIDUQL,indexSTID,indexETID)
		runningMetric1 = runningMetric(self.testIDNum)
		while self.testIDNum > 0: 
			trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i,modelfilName_i,datafilName_i,testTimeID_i,\
				testMetricF_i,testMetricR_i,testStockID_i = next(generator)		#generator.next() for python 2.x
			self.gbdtTrained_i,self.gbdtTestScore_i = \
				modelTrainer(self.paraTuneMethod,trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i)\
					.gbdt(mission=mission,validation=cvargs,**kwargs)
			print('%s_%s_%s testScore:%s'%(self.__trainMethod,'gbdt',self.paraTuneMethod,self.gbdtTestScore_i))
			runningMetric1.modelRankIC(mission,testMetricF_i,testMetricR_i,self.gbdtTrained_i)
			fileAccess().modelStorer(self.gbdtTrained_i,self.__trainMethod,'gbdt',self.paraTuneMethod,self.ortho_method,modelfilName_i)


			if len(cvargs[0]) != 0:
				param_dict = cvargs[0]

				if param_dict.get('cross_val_score'):
					cv = param_dict['cross_val_score'].get('cv',4)

					if mission == 'classification':
						scoring = param_dict['cross_val_score'].get('cross_val_score','roc_auc')
						modelValidator(self.gbdtTrained_i,self.trainF,self.trainL).cross_validation(cv,scoring)
					elif mission == 'regression':
						scoring = param_dict['cross_val_score'].get('cross_val_score','neg_mean_squared_error')
						modelValidator(self.gbdtTrained_i,self.trainF,self.trainR).cross_validation(cv,scoring)
					
				elif param_dict.get('time_series_score'):
					n_splits = param_dict['time_series_score'].get('n_splits',5)

					if mission == 'classification':
						modelValidator(self.gbdtTrained_i,self.trainF,self.trainL).time_series_validation(n_splits)
					elif mission == 'regression':
						modelValidator(self.gbdtTrained_i,self.trainF,self.trainR).time_series_validation(n_splits)

			self.testIDNum -= 1
			

	@timeLog('trailingMethod','total')
	def mlpc(self,mission='classification',*cvargs,**kwargs):
		timeIDUQL,indexSTID,indexETID = self.timeIndexInitial()
		generator = self.TrTeGenerator(timeIDUQL,indexSTID,indexETID)
		runningMetric1 = runningMetric(self.testIDNum)
		while self.testIDNum > 0: 
			trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i,modelfilName_i,datafilName_i,testTimeID_i,\
				testMetricF_i,testMetricR_i,testStockID_i = next(generator)		#generator.next() for python 2.x
			self.mlpcTrained_i,self.mlpcTestScore_i = \
				modelTrainer(self.paraTuneMethod,trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i)\
					.mlpc(mission=mission,validation=cvargs,**kwargs)
			print('%s_%s_%s testScore:%s'%(self.__trainMethod,'mlpc',self.paraTuneMethod,self.mlpcTestScore_i))
			runningMetric1.modelRankIC(mission,testMetricF_i,testMetricR_i,self.mlpcTrained_i)
			fileAccess().modelStorer(self.mlpcTrained_i,self.__trainMethod,'mlpc',self.paraTuneMethod,self.ortho_method,modelfilName_i)


			if len(cvargs[0]) != 0:
				param_dict = cvargs[0]

				if param_dict.get('cross_val_score'):
					cv = param_dict['cross_val_score'].get('cv',4)

					if mission == 'classification':
						scoring = param_dict['cross_val_score'].get('cross_val_score','roc_auc')
						modelValidator(self.mlpcTrained_i,self.trainF,self.trainL).cross_validation(cv,scoring)
					elif mission == 'regression':
						scoring = param_dict['cross_val_score'].get('cross_val_score','neg_mean_squared_error')
						modelValidator(self.mlpcTrained_i,self.trainF,self.trainR).cross_validation(cv,scoring)
					
				elif param_dict.get('time_series_score'):
					n_splits = param_dict['time_series_score'].get('n_splits',5)

					if mission == 'classification':
						modelValidator(self.mlpcTrained_i,self.trainF,self.trainL).time_series_validation(n_splits)
					elif mission == 'regression':
						modelValidator(self.mlpcTrained_i,self.trainF,self.trainR).time_series_validation(n_splits)

			self.testIDNum -= 1
			

	@timeLog('trailingMethod','total')
	def gnb(self,mission='classification',*cvargs,**kwargs):
		timeIDUQL,indexSTID,indexETID = self.timeIndexInitial()
		generator = self.TrTeGenerator(timeIDUQL,indexSTID,indexETID)
		runningMetric1 = runningMetric(self.testIDNum)
		while self.testIDNum > 0: 
			trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i,modelfilName_i,datafilName_i,testTimeID_i,\
				testMetricF_i,testMetricR_i,testStockID_i = next(generator)		#generator.next() for python 2.x
			self.gnbTrained_i,self.gnbTestScore_i = \
				modelTrainer(self.paraTuneMethod,trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i)\
					.gnb(mission=mission,validation=cvargs,**kwargs)
			print('%s_%s_%s testScore:%s'%(self.__trainMethod,'gnb',self.paraTuneMethod,self.gnbTestScore_i))
			runningMetric1.modelRankIC(mission,testMetricF_i,testMetricR_i,self.gnbTrained_i)
			fileAccess().modelStorer(self.gnbTrained_i,self.__trainMethod,'gnb',self.paraTuneMethod,self.ortho_method,modelfilName_i)


			if len(cvargs[0]) != 0:
				param_dict = cvargs[0]

				if param_dict.get('cross_val_score'):
					cv = param_dict['cross_val_score'].get('cv',4)
					scoring = param_dict['cross_val_score'].get('cross_val_score','roc_auc')
					modelValidator(self.gnbTrained_i,trainF_i,trainL_i).cross_validation(cv,scoring)

				elif param_dict.get('time_series_score'):
					n_splits = param_dict['time_series_score'].get('n_splits',5)
					modelValidator(self.gnbTrained_i,trainF_i,trainR_i).time_series_validation(n_splits)

			self.testIDNum -= 1
			


	@timeLog('trailingMethod','total')
	def adaboost(self,mission='classification',*cvargs,**kwargs):
		'''
		*cvargs: validation in/out sample
		default :cvNum=5,testSize=0.3
		**kwargs: args dict to model  
		'''
		timeIDUQL,indexSTID,indexETID = self.timeIndexInitial()
		generator = self.TrTeGenerator(timeIDUQL,indexSTID,indexETID)
		runningMetric1 = runningMetric(self.testIDNum)
		while self.testIDNum > 0: 
			trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i,modelfilName_i,datafilName_i,testTimeID_i,\
				testMetricF_i,testMetricR_i,testStockID_i = next(generator)		#generator.next() for python 2.x
			self.adaTrained_i,self.adaTestScore_i = \
				modelTrainer(self.paraTuneMethod,trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i)\
					.adaboost(mission=mission,validation=cvargs,**kwargs)
			print('%s_%s_%s testScore:%s'%(self.__trainMethod,'adaboost',self.paraTuneMethod,self.adaTestScore_i))
			runningMetric1.modelRankIC(mission,testMetricF_i,testMetricR_i,self.adaTrained_i)
			fileAccess().modelStorer(self.adaTrained_i,self.__trainMethod,'adaboost',self.paraTuneMethod,self.ortho_method,modelfilName_i)


			if len(cvargs[0]) != 0:
				param_dict = cvargs[0]

				if param_dict.get('cross_val_score'):
					cv = param_dict['cross_val_score'].get('cv',4)

					if mission == 'classification':
						scoring = param_dict['cross_val_score'].get('cross_val_score','roc_auc')
						modelValidator(self.adaTrained_i,trainF_i,trainL_i).cross_validation(cv,scoring)
					elif mission == 'regression':
						scoring = param_dict['cross_val_score'].get('cross_val_score','neg_mean_squared_error')
						modelValidator(self.adaTrained_i,trainF_i,trainR_i).cross_validation(cv,scoring)
					
				elif param_dict.get('time_series_score'):
					n_splits = param_dict['time_series_score'].get('n_splits',5)

					if mission == 'classification':
						modelValidator(self.adaTrained_i,trainF_i,trainL_i).time_series_validation(n_splits)
					elif mission == 'regression':
						modelValidator(self.adaTrained_i,trainF_i,trainR_i).time_series_validation(n_splits)
			
			self.testIDNum -= 1
			

	@timeLog('trailingMethod','total')
	def xgboost(self,mission='classification',*cvargs,**kwargs):
		timeIDUQL,indexSTID,indexETID = self.timeIndexInitial()
		generator = self.TrTeGenerator(timeIDUQL,indexSTID,indexETID)
		runningMetric1 = runningMetric(self.testIDNum)
		while self.testIDNum > 0: 
			trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i,modelfilName_i,datafilName_i,testTimeID_i,\
				testMetricF_i,testMetricR_i,testStockID_i = next(generator)		#generator.next() for python 2.x
			self.xgbTrained_i,self.xgbTestScore_i = \
				modelTrainer(self.paraTuneMethod,trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i)\
					.xgboost(mission=mission,validation=cvargs,**kwargs)
			print('%s_%s_%s testScore:%s'%(self.__trainMethod,'xgboost',self.paraTuneMethod,self.xgbTestScore_i))
			runningMetric1.modelRankIC(mission,testMetricF_i,testMetricR_i,self.xgbTrained_i)
			fileAccess().modelStorer(self.xgbTrained_i,self.__trainMethod,'xgboost',self.paraTuneMethod,self.ortho_method,modelfilName_i)


			if len(cvargs[0]) != 0:
				param_dict = cvargs[0]

				if param_dict.get('cross_val_score'):
					cv = param_dict['cross_val_score'].get('cv',4)

					if mission == 'classification':
						scoring = param_dict['cross_val_score'].get('cross_val_score','roc_auc')
						modelValidator(self.xgbTrained_i,trainF_i,trainL_i).cross_validation(cv,scoring)
					elif mission == 'regression':
						scoring = param_dict['cross_val_score'].get('cross_val_score','neg_mean_squared_error')
						modelValidator(self.xgbTrained_i,trainF_i,trainR_i).cross_validation(cv,scoring)
					
				elif param_dict.get('time_series_score'):
					n_splits = param_dict['time_series_score'].get('n_splits',5)

					if mission == 'classification':
						modelValidator(self.xgbTrained_i,trainF_i,trainL_i).time_series_validation(n_splits)
					elif mission == 'regression':
						modelValidator(self.xgbTrained_i,trainF_i,trainR_i).time_series_validation(n_splits)

			self.testIDNum -= 1
			

	@timeLog('trailingMethod','total')
	def lightgbm(self,mission='classification',*cvargs,**kwargs):
		timeIDUQL,indexSTID,indexETID = self.timeIndexInitial()
		generator = self.TrTeGenerator(timeIDUQL,indexSTID,indexETID)
		runningMetric1 = runningMetric(self.testIDNum)
		while self.testIDNum > 0: 
			trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i,modelfilName_i,datafilName_i,testTimeID_i,\
				testMetricF_i,testMetricR_i,testStockID_i = next(generator)		#generator.next() for python 2.x
			self.lgbTrained_i,self.lgbTestScore_i = \
				modelTrainer(self.paraTuneMethod,trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i)\
					.lightgbm(mission=mission,validation=cvargs,**kwargs)
			print('%s_%s_%s testScore:%s'%(self.__trainMethod,'lightgbm',self.paraTuneMethod,self.lgbTestScore_i))
			runningMetric1.modelRankIC(mission,testMetricF_i,testMetricR_i,self.lgbTrained_i)
			fileAccess().modelStorer(self.lgbTrained_i,self.__trainMethod,'lightgbm',self.paraTuneMethod,self.ortho_method,modelfilName_i)


			if len(cvargs[0]) != 0:
				param_dict = cvargs[0]

				if param_dict.get('cross_val_score'):
					cv = param_dict['cross_val_score'].get('cv',4)

					if mission == 'classification':
						scoring = param_dict['cross_val_score'].get('cross_val_score','roc_auc')
						modelValidator(self.lgbTrained_i,trainF_i,trainL_i).cross_validation(cv,scoring)
					elif mission == 'regression':
						scoring = param_dict['cross_val_score'].get('cross_val_score','neg_mean_squared_error')
						modelValidator(self.lgbTrained_i,trainF_i,trainR_i).cross_validation(cv,scoring)
					
				elif param_dict.get('time_series_score'):
					n_splits = param_dict['time_series_score'].get('n_splits',5)

					if mission == 'classification':
						modelValidator(self.lgbTrained_i,trainF_i,trainL_i).time_series_validation(n_splits)
					elif mission == 'regression':
						modelValidator(self.lgbTrained_i,trainF_i,trainR_i).time_series_validation(n_splits)
			
			self.testIDNum -= 1
			
	
	@timeLog('trailingMethod','total')
	def catboost(self,mission='classification',*cvargs,**kwargs):
		timeIDUQL,indexSTID,indexETID = self.timeIndexInitial()
		generator = self.TrTeGenerator(timeIDUQL,indexSTID,indexETID)
		runningMetric1 = runningMetric(self.testIDNum)
		while self.testIDNum > 0: 
			trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i,modelfilName_i,datafilName_i,testTimeID_i,\
				testMetricF_i,testMetricR_i,testStockID_i = next(generator)		#generator.next() for python 2.x
			self.cbTrained_i,self.cbTestScore_i = \
				modelTrainer(self.paraTuneMethod,trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i)\
					.catboost(mission=mission,validation=cvargs,**kwargs)
			print('%s_%s_%s testScore:%s'%(self.__trainMethod,'catboost',self.paraTuneMethod,self.cbTestScore_i))
			runningMetric1.modelRankIC(mission,testMetricF_i,testMetricR_i,self.cbTrained_i)
			fileAccess().modelStorer(self.cbTrained_i,self.__trainMethod,'catboost',self.paraTuneMethod,self.ortho_method,modelfilName_i)

			
			if len(cvargs[0]) != 0:
				param_dict = cvargs[0]

				if param_dict.get('cross_val_score'):
					cv = param_dict['cross_val_score'].get('cv',4)

					if mission == 'classification':
						scoring = param_dict['cross_val_score'].get('cross_val_score','roc_auc')
						modelValidator(self.cbTrained_i,trainF_i,trainL_i).cross_validation(cv,scoring)
					elif mission == 'regression':
						scoring = param_dict['cross_val_score'].get('cross_val_score','neg_mean_squared_error')
						modelValidator(self.cbTrained_i,trainF_i,trainR_i).cross_validation(cv,scoring)
					
				elif param_dict.get('time_series_score'):
					n_splits = param_dict['time_series_score'].get('n_splits',5)

					if mission == 'classification':
						modelValidator(self.cbTrained_i,trainF_i,trainL_i).time_series_validation(n_splits)
					elif mission == 'regression':
						modelValidator(self.cbTrained_i,trainF_i,trainR_i).time_series_validation(n_splits)

			self.testIDNum -= 1
			

	@timeLog('trailingMethod','total')
	def svm(self,mission='classification',*cvargs,**kwargs):
		timeIDUQL,indexSTID,indexETID = self.timeIndexInitial()
		generator = self.TrTeGenerator(timeIDUQL,indexSTID,indexETID)
		runningMetric1 = runningMetric(self.testIDNum)
		while self.testIDNum > 0: 
			trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i,modelfilName_i,datafilName_i,testTimeID_i,\
				testMetricF_i,testMetricR_i,testStockID_i = next(generator)		#generator.next() for python 2.x
			self.svmTrained_i,self.svmTestScore_i = \
				modelTrainer(self.paraTuneMethod,trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i)\
					.svm(mission=mission,validation=cvargs,**kwargs)
			print('%s_%s_%s testScore:%s'%(self.__trainMethod,'svm',self.paraTuneMethod,self.svmTestScore_i))
			runningMetric1.modelRankIC(mission,testMetricF_i,testMetricR_i,self.svmTrained_i)
			fileAccess().modelStorer(self.svmTrained_i,self.__trainMethod,'svm',self.paraTuneMethod,self.ortho_method,modelfilName_i)

			
			if len(cvargs[0]) != 0:
				param_dict = cvargs[0]

				if param_dict.get('cross_val_score'):
					cv = param_dict['cross_val_score'].get('cv',4)
					scoring = param_dict['cross_val_score'].get('cross_val_score','roc_auc')
					modelValidator(self.svmTrained_i,trainF_i,trainR_i).cross_validation(cv,scoring)

				elif param_dict.get('time_series_score'):
					n_splits = param_dict['time_series_score'].get('n_splits',5)
					modelValidator(self.svmTrained_i,trainF_i,trainR_i).time_series_validation(n_splits)

			self.testIDNum -= 1
			


	@timeLog('trailingMethod','total')
	def logistic(self,mission='classification',*cvargs,**kwargs):
		timeIDUQL,indexSTID,indexETID = self.timeIndexInitial()
		generator = self.TrTeGenerator(timeIDUQL,indexSTID,indexETID)
		runningMetric1 = runningMetric(self.testIDNum)
		while self.testIDNum > 0: 
			trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i,modelfilName_i,datafilName_i,testTimeID_i,\
				testMetricF_i,testMetricR_i,testStockID_i = next(generator)		#generator.next() for python 2.x
			self.logisticTrained_i,self.logisticTestScore_i = \
				modelTrainer(self.paraTuneMethod,trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i)\
					.logistic(mission=mission,validation=cvargs,**kwargs)
			print('%s_%s_%s testScore:%s'%(self.__trainMethod,'logistic',self.paraTuneMethod,self.logisticTestScore_i))
			runningMetric1.modelRankIC(mission,testMetricF_i,testMetricR_i,self.logisticTrained_i)
			fileAccess().modelStorer(self.logisticTrained_i,self.__trainMethod,'logistic',self.paraTuneMethod,self.ortho_method,modelfilName_i)

			
			if len(cvargs[0]) != 0:
				param_dict = cvargs[0]

				if param_dict.get('cross_val_score'):
					cv = param_dict['cross_val_score'].get('cv',4)
					scoring = param_dict['cross_val_score'].get('cross_val_score','roc_auc')
					modelValidator(self.logisticTrained_i,trainF_i,trainR_i).cross_validation(cv,scoring)

				elif param_dict.get('time_series_score'):
					n_splits = param_dict['time_series_score'].get('n_splits',5)
					modelValidator(self.logisticTrained_i,trainF_i,trainR_i).time_series_validation(n_splits)

			self.testIDNum -= 1
			

	@timeLog('trailingMethod','total')
	def linearSVC(self,mission='classification',*cvargs,**kwargs):
		timeIDUQL,indexSTID,indexETID = self.timeIndexInitial()
		generator = self.TrTeGenerator(timeIDUQL,indexSTID,indexETID)
		runningMetric1 = runningMetric(self.testIDNum)
		while self.testIDNum > 0: 
			trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i,modelfilName_i,datafilName_i,testTimeID_i,\
				testMetricF_i,testMetricR_i,testStockID_i = next(generator)		#generator.next() for python 2.x
			self.linearSVCTrained_i,self.linearSVCTestScore_i = \
				modelTrainer(self.paraTuneMethod,trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i)\
					.linearSVC(mission=mission,validation=cvargs,**kwargs)
			print('%s_%s_%s testScore:%s'%(self.__trainMethod,'linearsvc',self.paraTuneMethod,self.linearSVCTestScore_i))
			runningMetric1.modelRankIC(mission,testMetricF_i,testMetricR_i,self.linearSVCTrained_i)
			fileAccess().modelStorer(self.linearSVCTrained_i,self.__trainMethod,'linearsvc',self.paraTuneMethod,self.ortho_method,modelfilName_i)

			
			if len(cvargs[0]) != 0:
				param_dict = cvargs[0]

				if param_dict.get('cross_val_score'):
					cv = param_dict['cross_val_score'].get('cv',4)
					scoring = param_dict['cross_val_score'].get('cross_val_score','roc_auc')
					modelValidator(self.linearSVCTrained_i,trainF_i,trainR_i).cross_validation(cv,scoring)

				elif param_dict.get('time_series_score'):
					n_splits = param_dict['time_series_score'].get('n_splits',5)
					modelValidator(self.linearSVCTrained_i,trainF_i,trainR_i).time_series_validation(n_splits)

			self.testIDNum -= 1
			

	@timeLog('trailingMethod','total')
	def sgd(self,mission='classification',*cvargs,**kwargs):
		timeIDUQL,indexSTID,indexETID = self.timeIndexInitial()
		generator = self.TrTeGenerator(timeIDUQL,indexSTID,indexETID)
		runningMetric1 = runningMetric(self.testIDNum)
		while self.testIDNum > 0: 
			trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i,modelfilName_i,datafilName_i,testTimeID_i,\
				testMetricF_i,testMetricR_i,testStockID_i = next(generator)		#generator.next() for python 2.x
			self.sgdTrained_i,self.sgdTestScore_i = \
				modelTrainer(self.paraTuneMethod,trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i)\
					.sgd(mission=mission,validation=cvargs,**kwargs)
			print('%s_%s_%s testScore:%s'%(self.__trainMethod,'sgd',self.paraTuneMethod,self.sgdTestScore_i))
			runningMetric1.modelRankIC(mission,testMetricF_i,testMetricR_i,self.sgdTrained_i)
			fileAccess().modelStorer(self.sgdTrained_i,self.__trainMethod,'sgd',self.paraTuneMethod,self.ortho_method,modelfilName_i)

			
			if len(cvargs[0]) != 0:
				param_dict = cvargs[0]

				if param_dict.get('cross_val_score'):
					cv = param_dict['cross_val_score'].get('cv',4)

					if mission == 'classification':
						scoring = param_dict['cross_val_score'].get('cross_val_score','roc_auc')
						modelValidator(self.sgdTrained_i,trainF_i,trainL_i).cross_validation(cv,scoring)
					elif mission == 'regression':
						scoring = param_dict['cross_val_score'].get('cross_val_score','neg_mean_squared_error')
						modelValidator(self.sgdTrained_i,trainF_i,trainR_i).cross_validation(cv,scoring)
					
				elif param_dict.get('time_series_score'):
					n_splits = param_dict['time_series_score'].get('n_splits',5)

					if mission == 'classification':
						modelValidator(self.sgdTrained_i,trainF_i,trainL_i).time_series_validation(n_splits)
					elif mission == 'regression':
						modelValidator(self.sgdTrained_i,trainF_i,trainR_i).time_series_validation(n_splits)

			self.testIDNum -= 1


	@timeLog('trailingMethod','total')
	def deepforest(self,mission='classification',*cvargs,**kwargs):
		timeIDUQL,indexSTID,indexETID = self.timeIndexInitial()
		generator = self.TrTeGenerator(timeIDUQL,indexSTID,indexETID)
		runningMetric1 = runningMetric(self.testIDNum)
		while self.testIDNum > 0: 
			trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i,modelfilName_i,datafilName_i,testTimeID_i,\
				testMetricF_i,testMetricR_i,testStockID_i = next(generator)		#generator.next() for python 2.x
			self.dfTrained_i,self.dfTestScore_i = \
				modelTrainer(self.paraTuneMethod,trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i)\
					.deepforest(mission=mission,validation=cvargs,**kwargs)
			print('%s_%s_%s testScore:%s'%(self.__trainMethod,'deepforest',self.paraTuneMethod,self.dfTestScore_i))
			runningMetric1.modelRankIC(mission,testMetricF_i.values,testMetricR_i.values,self.dfTrained_i)
			fileAccess().modelStorer(self.dfTrained_i,self.__trainMethod,'deepforest',self.paraTuneMethod,self.ortho_method,modelfilName_i)

			
			if len(cvargs[0]) != 0:
				param_dict = cvargs[0]

				if param_dict.get('cross_val_score'):
					cv = param_dict['cross_val_score'].get('cv',4)

					if mission == 'classification':
						scoring = param_dict['cross_val_score'].get('cross_val_score','roc_auc')
						modelValidator(self.dfTrained_i,trainF_i,trainL_i).cross_validation(cv,scoring)
					elif mission == 'regression':
						scoring = param_dict['cross_val_score'].get('cross_val_score','neg_mean_squared_error')
						modelValidator(self.dfTrained_i,trainF_i,trainR_i).cross_validation(cv,scoring)
					
				elif param_dict.get('time_series_score'):
					n_splits = param_dict['time_series_score'].get('n_splits',5)

					if mission == 'classification':
						modelValidator(self.dfTrained_i,trainF_i,trainL_i).time_series_validation(n_splits)
					elif mission == 'regression':
						modelValidator(self.dfTrained_i,trainF_i,trainR_i).time_series_validation(n_splits)

			self.testIDNum -= 1
			

# cyclical trailing method
class cyclicalTrailMethod(data2Model):
	'''
	cyclical trailing train with trainWindowL timeIDs as train window(e.g. all trainWindowL januarys to train for next year january test)
	startTrainTID: the first trailing train timeID. So the first train window will be [startTrainTID,startTrainTID+trainWindowL-1]
	'''
	__trainMethod = 'cyclicalTrailingMethod'

	def __init__(self,TrTeDF,modelMetricDF,startTestTID,ortho_method='none',num_category=5,trainWindowL=7,window=1,paraTuneMethod='NoHyperTune',**model_dict):
		super(cyclicalTrailMethod,self).__init__(TrTeDF,modelMetricDF,paraTuneMethod)
		self.startTestTID = startTestTID
		self.num_category = num_category
		self.trainWindowL = trainWindowL
		self.window = window
		self.paraTuneMethod = paraTuneMethod
		self.testIDNum = None
		self.model = model_dict
		self.ortho_method = ortho_method

	# get index of time data
	def timeIndexInitial(self):
		self.rank = self.timeID.rank(method='dense').astype(int)
		self.rank = self.rank.apply(lambda x: x % self.num_category).rename('rank')
		self.rank = pd.concat([self.rank,self.timeID],axis=1)
		self.rank.drop_duplicates(subset=['timeID'],inplace=True)
		self.rank = np.array(self.rank['rank'].tolist())
		# get the time id for the data
		timeIDUQ = pd.unique(self.timeID)
		timeIDUQL = timeIDUQ.tolist()
		# get the index of start date in the list
		try:
			indexSTID = timeIDUQL.index(self.startTestTID)	#start test time id 
		except:
			indexSTID = sorted(timeIDUQL+[self.startTestTID]).index(self.startTestTID)
			self.startTestTID = timeIDUQL[indexSTID]
		#indexETID = indexSTID + (self.trainWindowL-1)

		# get the data from the start date
		testTimeIDL = timeIDUQL[indexSTID:]		#rest test time id 
		self.testIDNum = len(range(0,len(testTimeIDL),self.window))
		print('%s timeIDs to test'%self.testIDNum)
		return testTimeIDL,timeIDUQL,timeIDUQ

	# Split the train and test data
	def TrTeGenerator(self,testTimeIDL,timeIDUQL,timeIDUQ):
		
		for i in range(0,len(testTimeIDL),self.window):
			# get the test data at time i
			print('Test timeID:%s'%testTimeIDL[i])
			testF_i = self.feature[self.timeID==testTimeIDL[i]]
			testL_i = self.label[self.timeID==testTimeIDL[i]]
			testR_i = self.yieldRate[self.timeID==testTimeIDL[i]]	
			category_i = self.rank[testTimeIDL.index(testTimeIDL[i])]
			testMetricF_i = self.feature_metric[self.timeID_metric==testTimeIDL[i]]
			testMetricR_i = self.yieldRate_metric[self.timeID_metric==testTimeIDL[i]]
			testStockID_i = self.stockID_metric[self.timeID_metric==testTimeIDL[i]]

			indexTest_i = timeIDUQL.index(testTimeIDL[i])
			indexMsID_i = np.where(self.rank==category_i)[0]	#indexs of months for i 
			# get the training data, which is the data from the same month in last few years
			trainTimeIDLst_i = timeIDUQ[indexMsID_i[indexMsID_i<indexTest_i][-self.trainWindowL:]]
			print('Train timeID:[%s->%s]. %s timeIDs to train'%(trainTimeIDLst_i[0],trainTimeIDLst_i[-1],len(trainTimeIDLst_i)))

			modelfilName_i = testTimeIDL[i] + '.m'
			datafilName_i = testTimeIDL[i] + '.pkl'

			# use append to get the whole data
			for j in trainTimeIDLst_i:

				if j == trainTimeIDLst_i[0]:				
					trainF_i = self.feature[self.timeID==j]
					trainL_i = self.label[self.timeID==j]
					trainR_i = self.yieldRate[self.timeID==j]
					trainTimeID_i = self.timeID[self.timeID==j]
					trainStockID_i = self.stockID[self.timeID==j]
					
				else:
					trainF_i = trainF_i.append(self.feature[self.timeID==j])
					trainL_i = trainL_i.append(self.label[self.timeID==j])
					trainR_i = trainR_i.append(self.yieldRate[self.timeID==j])

			# return the generator of the train and test data
			yield (trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i,\
				modelfilName_i,datafilName_i,testMetricF_i,testMetricR_i,testStockID_i)

	def set_hyperopt_params(self,**kwargs):
		for key,value in kwargs.items():
			if self.model.get(key):
				self.model[key].update({'hyperopt':value})
				if not self.model[key].get('validation'):
					self.model[key].update({'validation':{}})
			else:
				self.model.update({key:{'hyperopt':value,'validation':{}}})

	def set_validation_params(self,**kwargs):
		for key,value in kwargs.items():
			if self.model.get(key):
				self.model[key].update({'validation':value})
				if not self.model[key].get('hyperopt'):
					self.model[key].update({'hyperopt':{}})
			else:
				self.model.update({key:{'hyperopt':{},'validation':value}})

	def runModels(self):
		
		if not self.model:
			self.kernelRidge()
			self.ridge()
			self.lasso()
			self.elasticNet()
			self.svr()
			self.randomForest()
			self.gbdt()
			self.mlpc()
			self.gnb()
			self.adaboost()
			self.xgboost()
			self.lightgbm()
			self.catboost()
			self.svm()
			self.logistic()
			self.linearSVC()
			self.sgd()
			self.deepforest()
		else:
			if not self.model.get('model'):
				raise KeyError('key "model" should be in the model dictionary!')
			if not self.model.get('mission'):
				raise KeyError('key "mission" should be in the model dictionary!')
			if len(self.model['mission']) == 1:
				self.model.update({'mission':self.model['mission']*len(self.model['model'])})
			model_lst = self.model['model']
			mission_lst = self.model['mission']
			for i in range(len(model_lst)):
				if model_lst[i] == 'kernelridge':
					if self.model.get(model_lst[i]):
						self.kernelRidge(mission_lst[i],*[self.model[model_lst[i]]['validation']],**self.model[model_lst[i]]['hyperopt'])
					else:
						self.kernelRidge(mission=mission_lst[i])
				elif model_lst[i] == 'ridge':
					if self.model.get(model_lst[i]):
						self.ridge(mission_lst[i],*[self.model[model_lst[i]]['validation']],**self.model[model_lst[i]]['hyperopt'])
					else:
						self.ridge(mission=mission_lst[i])
				elif model_lst[i] == 'lasso':
					if self.model.get(model_lst[i]):
						self.lasso(mission_lst[i],*[self.model[model_lst[i]]['validation']],**self.model[model_lst[i]]['hyperopt'])
					else:
						self.lasso(mission=mission_lst[i])
				elif model_lst[i] == 'elasticnet':
					if self.model.get(model_lst[i]):
						self.elasticNet(mission_lst[i],*[self.model[model_lst[i]]['validation']],**self.model[model_lst[i]]['hyperopt'])
					else:
						self.elasticNet(mission=mission_lst[i])
				elif model_lst[i] == 'svr':
					if self.model.get(model_lst[i]):
						self.svr(mission_lst[i],*[self.model[model_lst[i]]['validation']],**self.model[model_lst[i]]['hyperopt'])
					else:
						self.svr(mission=mission_lst[i])
				elif model_lst[i] == 'randomforest':
					if self.model.get(model_lst[i]):
						self.randomForest(mission_lst[i],*[self.model[model_lst[i]]['validation']],**self.model[model_lst[i]]['hyperopt'])
					else:
						self.randomForest(mission=mission_lst[i])
				elif model_lst[i] == 'gbdt':
					if self.model.get(model_lst[i]):
						self.gbdt(mission_lst[i],*[self.model[model_lst[i]]['validation']],**self.model[model_lst[i]]['hyperopt'])
					else:
						self.gbdt(mission=mission_lst[i])
				elif model_lst[i] == 'mlpc':
					if self.model.get(model_lst[i]):
						self.mlpc(mission_lst[i],*[self.model[model_lst[i]]['validation']],**self.model[model_lst[i]]['hyperopt'])
					else:
						self.mlpc(mission=mission_lst[i])
				elif model_lst[i] == 'gnb':
					if self.model.get(model_lst[i]):
						self.gnb(mission_lst[i],*[self.model[model_lst[i]]['validation']],**self.model[model_lst[i]]['hyperopt'])
					else:
						self.gnb(mission=mission_lst[i])
				elif model_lst[i] == 'adaboost':
					if self.model.get(model_lst[i]):
						self.adaboost(mission_lst[i],*[self.model[model_lst[i]]['validation']],**self.model[model_lst[i]]['hyperopt'])
					else:
						self.adaboost(mission=mission_lst[i])
				elif model_lst[i] == 'xgboost':
					if self.model.get(model_lst[i]):
						self.xgboost(mission_lst[i],*[self.model[model_lst[i]]['validation']],**self.model[model_lst[i]]['hyperopt'])
					else:
						self.xgboost(mission=mission_lst[i])
				elif model_lst[i] == 'lightgbm':
					if self.model.get(model_lst[i]):
						self.lightgbm(mission_lst[i],*[self.model[model_lst[i]]['validation']],**self.model[model_lst[i]]['hyperopt'])
					else:
						self.lightgbm(mission=mission_lst[i])
				elif model_lst[i] == 'catboost':
					if self.model.get(model_lst[i]):
						self.catboost(mission_lst[i],*[self.model[model_lst[i]]['validation']],**self.model[model_lst[i]]['hyperopt'])
					else:
						self.catboost(mission=mission_lst[i])
				elif model_lst[i] == 'svm':
					if self.model.get(model_lst[i]):
						self.svm(mission_lst[i],*[self.model[model_lst[i]]['validation']],**self.model[model_lst[i]]['hyperopt'])
					else:
						self.svm(mission=mission_lst[i])
				elif model_lst[i] == 'logistic':
					if self.model.get(model_lst[i]):
						self.logistic(mission_lst[i],*[self.model[model_lst[i]]['validation']],**self.model[model_lst[i]]['hyperopt'])
					else:
						self.logistic(mission=mission_lst[i])
				elif model_lst[i] == 'linearsvc':
					if self.model.get(model_lst[i]):
						self.linearSVC(mission_lst[i],*[self.model[model_lst[i]]['validation']],**self.model[model_lst[i]]['hyperopt'])
					else:
						self.linearSVC(mission=mission_lst[i])
				elif model_lst[i] == 'sgd':
					if self.model.get(model_lst[i]):
						self.sgd(mission_lst[i],*[self.model[model_lst[i]]['validation']],**self.model[model_lst[i]]['hyperopt'])
					else:
						self.sgd(mission=mission_lst[i])
				elif model_lst[i] == 'deepforest':
					if self.model.get(model_lst[i]):
						self.deepforest(mission_lst[i],*[self.model[model_lst[i]]['validation']],**self.model[model_lst[i]]['hyperopt'])
					else:
						self.deepforest(mission=mission_lst[i])


	@timeLog('cyclicalTrailingMethod','total')
	def kernelRidge(self,mission='regression',*cvargs,**kwargs):
		'''
		*cvargs: validation in/out sample
		default :cvNum=5,testSize=0.3
		**kwargs: args dict to model  
		'''
		# get the initial index for time id
		testTimeIDL,timeIDUQL,timeIDUQ = self.timeIndexInitial()
		# get generator that generates training data and test data
		generator = self.TrTeGenerator(testTimeIDL,timeIDUQL,timeIDUQ)
		runningMetric_ = runningMetric(self.testIDNum)
		while self.testIDNum > 0: 
			trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i,modelfilName_i,datafilName_i,\
				testMetricF_i,testMetricR_i,testStockID_i = next(generator)		#generator.next() for python 2.x
			self.kernelRidgeTrained_i,self.kernelRidgeTestScore_i = \
				modelTrainer(self.paraTuneMethod,trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i)\
					.kernelRidge(mission=mission,**kwargs)
			print('%s_%s_%s testScore:%s'%(self.__trainMethod,'kernelRidge',self.paraTuneMethod,self.kernelRidgeTestScore_i))
			runningMetric_.modelRankIC(mission,testF_i,testR_i,self.kernelRidgeTrained_i)
			fileAccess().modelStorer(self.kernelRidgeTrained_i,self.__trainMethod,'kernelRidge',self.paraTuneMethod,self.ortho_method,modelfilName_i)


		if len(cvargs[0]) != 0:
			param_dict = cvargs[0]

			if param_dict.get('cross_val_score'):
				cv = param_dict['cross_val_score'].get('cv',4)
				scoring = param_dict['cross_val_score'].get('cross_val_score','neg_mean_squared_error')
				modelValidator(self.kernelRidgeTrained_i,trainF_i,trainR_i).cross_validation(cv,scoring)

			elif param_dict.get('time_series_score'):
				n_splits = param_dict['time_series_score'].get('n_splits',5)
				modelValidator(self.kernelRidgeTrained_i,trainF_i,trainR_i).time_series_validation(n_splits)

			self.testIDNum -= 1
			


	@timeLog('cyclicalTrailingMethod','total')
	def ridge(self,mission='regression',*cvargs,**kwargs):
		'''
		*cvargs: validation in/out sample
		default :cvNum=5,testSize=0.3
		**kwargs: args dict to model  
		'''
		testTimeIDL,timeIDUQL,timeIDUQ = self.timeIndexInitial()
		generator = self.TrTeGenerator(testTimeIDL,timeIDUQL,timeIDUQ)
		runningMetric_ = runningMetric(self.testIDNum)
		while self.testIDNum > 0: 
			trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i,modelfilName_i,datafilName_i,\
				testMetricF_i,testMetricR_i,testStockID_i = next(generator)		#generator.next() for python 2.x
			self.ridgeTrained_i,self.ridgeTestScore_i = \
				modelTrainer(self.paraTuneMethod,trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i)\
					.ridge(mission=mission,**kwargs)
			print('%s_%s_%s testScore:%s'%(self.__trainMethod,'ridge',self.paraTuneMethod,self.ridgeTestScore_i))
			runningMetric_.modelRankIC(mission,testF_i,testR_i,self.ridgeTrained_i)
			fileAccess().modelStorer(self.ridgeTrained_i,self.__trainMethod,'ridge',self.paraTuneMethod,self.ortho_method,modelfilName_i)


			if len(cvargs[0]) != 0:
				param_dict = cvargs[0]

				if param_dict.get('cross_val_score'):
					cv = param_dict['cross_val_score'].get('cv',4)

					if mission == 'classification':
						scoring = param_dict['cross_val_score'].get('cross_val_score','roc_auc')
						modelValidator(self.ridgeTrained_i,trainF_i,trainL_i).cross_validation(cv,scoring)
					elif mission == 'regression':
						scoring = param_dict['cross_val_score'].get('cross_val_score','neg_mean_squared_error')
						modelValidator(self.ridgeTrained_i,trainF_i,trainR_i).cross_validation(cv,scoring)
					
				elif param_dict.get('time_series_score'):
					n_splits = param_dict['time_series_score'].get('n_splits',5)

					if mission == 'classification':
						modelValidator(self.ridgeTrained_i,trainF_i,trainL_i).time_series_validation(n_splits)
					elif mission == 'regression':
						modelValidator(self.ridgeTrained_i,trainF_i,trainR_i).time_series_validation(n_splits)

			self.testIDNum -= 1
			

	@timeLog('cyclicalTrailingMethod','total')
	def lasso(self,mission='regression',*cvargs,**kwargs):
		'''
		*cvargs: validation in/out sample
		default :cvNum=5,testSize=0.3
		**kwargs: args dict to model  
		'''
		testTimeIDL,timeIDUQL,timeIDUQ = self.timeIndexInitial()
		generator = self.TrTeGenerator(testTimeIDL,timeIDUQL,timeIDUQ)
		runningMetric_ = runningMetric(self.testIDNum)
		while self.testIDNum > 0: 
			trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i,modelfilName_i,datafilName_i,\
				testMetricF_i,testMetricR_i,testStockID_i = next(generator)		#generator.next() for python 2.x
			self.lassoTrained_i,self.lassoTestScore_i = \
				modelTrainer(self.paraTuneMethod,trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i)\
					.lasso(mission=mission,**kwargs)
			print('%s_%s_%s testScore:%s'%(self.__trainMethod,'lasso',self.paraTuneMethod,self.lassoTestScore_i))
			runningMetric_.modelRankIC(mission,testF_i,testR_i,self.lassoTrained_i)
			fileAccess().modelStorer(self.lassoTrained_i,self.__trainMethod,'lasso',self.paraTuneMethod,self.ortho_method,modelfilName_i)


			if len(cvargs[0]) != 0:
				param_dict = cvargs[0]

				if param_dict.get('cross_val_score'):
					cv = param_dict['cross_val_score'].get('cv',4)
					scoring = param_dict['cross_val_score'].get('cross_val_score','neg_mean_squared_error')
					modelValidator(self.lassoTrained_i,trainF_i,trainR_i).cross_validation(cv,scoring)

				elif param_dict.get('time_series_score'):
					n_splits = param_dict['time_series_score'].get('n_splits',5)
					modelValidator(self.lassoTrained_i,trainF_i,trainR_i).time_series_validation(n_splits)

			self.testIDNum -= 1
			

	@timeLog('cyclicalTrailingMethod','total')
	def elasticNet(self,mission='regression',*cvargs,**kwargs):
		'''
		*cvargs: validation in/out sample
		default :cvNum=5,testSize=0.3
		**kwargs: args dict to model  
		'''
		testTimeIDL,timeIDUQL,timeIDUQ = self.timeIndexInitial()
		generator = self.TrTeGenerator(testTimeIDL,timeIDUQL,timeIDUQ)
		runningMetric_ = runningMetric(self.testIDNum)
		while self.testIDNum > 0: 
			trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i,modelfilName_i,datafilName_i,\
				testMetricF_i,testMetricR_i,testStockID_i = next(generator)		#generator.next() for python 2.x
			self.elasticNetTrained_i,self.elasticNetTestScore_i = \
				modelTrainer(self.paraTuneMethod,trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i)\
					.elasticNet(mission=mission,**kwargs)
			print('%s_%s_%s testScore:%s'%(self.__trainMethod,'elasticNet',self.paraTuneMethod,self.elasticNetTestScore_i))
			runningMetric_.modelRankIC(mission,testF_i,testR_i,self.elasticNetTrained_i)
			fileAccess().modelStorer(self.elasticNetTrained_i,self.__trainMethod,'elasticNet',self.paraTuneMethod,self.ortho_method,modelfilName_i)


			if len(cvargs[0]) != 0:
				param_dict = cvargs[0]

				if param_dict.get('cross_val_score'):
					cv = param_dict['cross_val_score'].get('cv',4)
					scoring = param_dict['cross_val_score'].get('cross_val_score','neg_mean_squared_error')
					modelValidator(self.elasticNetTrained_i,trainF_i,trainR_i).cross_validation(cv,scoring)

				elif param_dict.get('time_series_score'):
					n_splits = param_dict['time_series_score'].get('n_splits',5)
					modelValidator(self.elasticNetTrained_i,trainF_i,trainR_i).time_series_validation(n_splits)

			self.testIDNum -= 1
			

	@timeLog('cyclicalTrailingMethod','total')
	def svr(self,mission='regression',*cvargs,**kwargs):
		'''
		*cvargs: validation in/out sample
		default :cvNum=5,testSize=0.3
		**kwargs: args dict to model  
		'''
		testTimeIDL,timeIDUQL,timeIDUQ = self.timeIndexInitial()
		generator = self.TrTeGenerator(testTimeIDL,timeIDUQL,timeIDUQ)
		runningMetric_ = runningMetric(self.testIDNum)
		while self.testIDNum > 0: 
			trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i,modelfilName_i,datafilName_i,\
				testMetricF_i,testMetricR_i,testStockID_i = next(generator)		#generator.next() for python 2.x
			self.svrTrained_i,self.svrTestScore_i = \
				modelTrainer(self.paraTuneMethod,trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i)\
					.svr(mission=mission,**kwargs)
			print('%s_%s_%s testScore:%s'%(self.__trainMethod,'svr',self.paraTuneMethod,self.svrTestScore_i))
			runningMetric_.modelRankIC(mission,testF_i,testR_i,self.svrTrained_i)
			fileAccess().modelStorer(self.svrTrained_i,self.__trainMethod,'svr',self.paraTuneMethod,self.ortho_method,modelfilName_i)


			if len(cvargs[0]) != 0:
				param_dict = cvargs[0]

				if param_dict.get('cross_val_score'):
					cv = param_dict['cross_val_score'].get('cv',4)
					scoring = param_dict['cross_val_score'].get('cross_val_score','neg_mean_squared_error')
					modelValidator(self.svrTrained_i,trainF_i,trainR_i).cross_validation(cv,scoring)

				elif param_dict.get('time_series_score'):
					n_splits = param_dict['time_series_score'].get('n_splits',5)
					modelValidator(self.svrTrained_i,trainF_i,trainR_i).time_series_validation(n_splits)

			self.testIDNum -= 1
			

	@timeLog('cyclicalTrailingMethod','total')
	def randomForest(self,mission='classification',*cvargs,**kwargs):
		'''
		*cvargs: validation in/out sample
		default :cvNum=5,testSize=0.3
		**kwargs: args dict to model  
		'''
		testTimeIDL,timeIDUQL,timeIDUQ = self.timeIndexInitial()
		generator = self.TrTeGenerator(testTimeIDL,timeIDUQL,timeIDUQ)
		runningMetric_ = runningMetric(self.testIDNum)
		while self.testIDNum > 0: 
			trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i,modelfilName_i,datafilName_i,\
				testMetricF_i,testMetricR_i,testStockID_i = next(generator)		#generator.next() for python 2.x
			self.randomForestTrained_i,self.randomForestTestScore_i = \
				modelTrainer(self.paraTuneMethod,trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i)\
					.randomForest(mission=mission,**kwargs)
			print('%s_%s_%s testScore:%s'%(self.__trainMethod,'randomforest',self.paraTuneMethod,self.randomForestTestScore_i))
			runningMetric_.modelRankIC(mission,testF_i,testR_i,self.randomForestTrained_i)
			fileAccess().modelStorer(self.randomForestTrained_i,self.__trainMethod,'randomforest',self.paraTuneMethod,self.ortho_method,modelfilName_i)


			if len(cvargs[0]) != 0:
				param_dict = cvargs[0]

				if param_dict.get('cross_val_score'):
					cv = param_dict['cross_val_score'].get('cv',4)

					if mission == 'classification':
						scoring = param_dict['cross_val_score'].get('cross_val_score','roc_auc')
						modelValidator(self.randomForestTrained_i,trainF_i,trainL_i).cross_validation(cv,scoring)
					elif mission == 'regression':
						scoring = param_dict['cross_val_score'].get('cross_val_score','neg_mean_squared_error')
						modelValidator(self.randomForestTrained_i,trainF_i,trainR_i).cross_validation(cv,scoring)
					
				elif param_dict.get('time_series_score'):
					n_splits = param_dict['time_series_score'].get('n_splits',5)

					if mission == 'classification':
						modelValidator(self.randomForestTrained_i,trainF_i,trainL_i).time_series_validation(n_splits)
					elif mission == 'regression':
						modelValidator(self.randomForestTrained_i,trainF_i,trainR_i).time_series_validation(n_splits)

			self.testIDNum -= 1
			

	@timeLog('cyclicalTrailingMethod','total')
	def gbdt(self,mission='classification',*cvargs,**kwargs):
		'''
		*cvargs: validation in/out sample
		default :cvNum=5,testSize=0.3
		**kwargs: args dict to model  
		'''
		testTimeIDL,timeIDUQL,timeIDUQ = self.timeIndexInitial()
		generator = self.TrTeGenerator(testTimeIDL,timeIDUQL,timeIDUQ)
		runningMetric_ = runningMetric(self.testIDNum)
		while self.testIDNum > 0: 
			trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i,modelfilName_i,datafilName_i,\
				testMetricF_i,testMetricR_i,testStockID_i = next(generator)		#generator.next() for python 2.x
			self.gbdtTrained_i,self.gbdtTestScore_i = \
				modelTrainer(self.paraTuneMethod,trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i)\
					.gbdt(mission=mission,**kwargs)
			print('%s_%s_%s testScore:%s'%(self.__trainMethod,'gbdt',self.paraTuneMethod,self.gbdtTestScore_i))
			runningMetric_.modelRankIC(mission,testF_i,testR_i,self.gbdtTrained_i)
			fileAccess().modelStorer(self.gbdtTrained_i,self.__trainMethod,'gbdt',self.paraTuneMethod,self.ortho_method,modelfilName_i)


			if len(cvargs[0]) != 0:
				param_dict = cvargs[0]

				if param_dict.get('cross_val_score'):
					cv = param_dict['cross_val_score'].get('cv',4)

					if mission == 'classification':
						scoring = param_dict['cross_val_score'].get('cross_val_score','roc_auc')
						modelValidator(self.gbdtTrained_i,self.trainF,self.trainL).cross_validation(cv,scoring)
					elif mission == 'regression':
						scoring = param_dict['cross_val_score'].get('cross_val_score','neg_mean_squared_error')
						modelValidator(self.gbdtTrained_i,self.trainF,self.trainR).cross_validation(cv,scoring)
					
				elif param_dict.get('time_series_score'):
					n_splits = param_dict['time_series_score'].get('n_splits',5)

					if mission == 'classification':
						modelValidator(self.gbdtTrained_i,self.trainF,self.trainL).time_series_validation(n_splits)
					elif mission == 'regression':
						modelValidator(self.gbdtTrained_i,self.trainF,self.trainR).time_series_validation(n_splits)

			self.testIDNum -= 1
			
	
	@timeLog('cyclicalTrailingMethod','total')
	def mlpc(self,mission='classification',*cvargs,**kwargs):
		'''
		*cvargs: validation in/out sample
		default :cvNum=5,testSize=0.3
		**kwargs: args dict to model  
		'''
		testTimeIDL,timeIDUQL,timeIDUQ = self.timeIndexInitial()
		generator = self.TrTeGenerator(testTimeIDL,timeIDUQL,timeIDUQ)
		runningMetric_ = runningMetric(self.testIDNum)
		while self.testIDNum > 0: 
			trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i,modelfilName_i,datafilName_i,\
				testMetricF_i,testMetricR_i,testStockID_i = next(generator)		#generator.next() for python 2.x
			self.mlpcTrained_i,self.mlpcTestScore_i = \
				modelTrainer(self.paraTuneMethod,trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i)\
					.mlpc(mission=mission,**kwargs)
			print('%s_%s_%s testScore:%s'%(self.__trainMethod,'mlpc',self.paraTuneMethod,self.mlpcTestScore_i))
			runningMetric_.modelRankIC(mission,testF_i,testR_i,self.mlpcTrained_i)
			fileAccess().modelStorer(self.mlpcTrained_i,self.__trainMethod,'mlpc',self.paraTuneMethod,self.ortho_method,modelfilName_i)


			if len(cvargs[0]) != 0:
				param_dict = cvargs[0]

				if param_dict.get('cross_val_score'):
					cv = param_dict['cross_val_score'].get('cv',4)


					if mission == 'classification':
						scoring = param_dict['cross_val_score'].get('cross_val_score','roc_auc')
						modelValidator(self.mlpcTrained_i,trainF_i,trainL_i).cross_validation(cv,scoring)
					elif mission == 'regression':
						scoring = param_dict['cross_val_score'].get('cross_val_score','neg_mean_squared_error')
						modelValidator(self.mlpcTrained_i,trainF_i,trainR_i).cross_validation(cv,scoring)
					
				elif param_dict.get('time_series_score'):
					n_splits = param_dict['time_series_score'].get('n_splits',5)

					if mission == 'classification':
						modelValidator(self.mlpcTrained_i,trainF_i,trainL_i).time_series_validation(n_splits)
					elif mission == 'regression':
						modelValidator(self.mlpcTrained_i,trainF_i,trainR_i).time_series_validation(n_splits)

			self.testIDNum -= 1
			

	@timeLog('cyclicalTrailingMethod','total')
	def gnb(self,mission='classification',*cvargs,**kwargs):
		'''
		*cvargs: validation in/out sample
		default :cvNum=5,testSize=0.3
		**kwargs: args dict to model  
		'''
		testTimeIDL,timeIDUQL,timeIDUQ = self.timeIndexInitial()
		generator = self.TrTeGenerator(testTimeIDL,timeIDUQL,timeIDUQ)
		runningMetric_ = runningMetric(self.testIDNum)
		while self.testIDNum > 0: 
			trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i,modelfilName_i,datafilName_i,\
				testMetricF_i,testMetricR_i,testStockID_i = next(generator)		#generator.next() for python 2.x
			self.gnbTrained_i,self.gnbTestScore_i = \
				modelTrainer(self.paraTuneMethod,trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i)\
					.gnb(mission=mission,**kwargs)
			print('%s_%s_%s testScore:%s'%(self.__trainMethod,'gnb',self.paraTuneMethod,self.gnbTestScore_i))
			runningMetric_.modelRankIC(mission,testF_i,testR_i,self.gnbTrained_i)
			fileAccess().modelStorer(self.gnbTrained_i,self.__trainMethod,'gnb',self.paraTuneMethod,self.ortho_method,modelfilName_i)


			if len(cvargs[0]) != 0:
				param_dict = cvargs[0]

				if param_dict.get('cross_val_score'):
					cv = param_dict['cross_val_score'].get('cv',4)
					scoring = param_dict['cross_val_score'].get('cross_val_score','roc_auc')
					modelValidator(self.gnbTrained_i,trainF_i,trainR_i).cross_validation(cv,scoring)

				elif param_dict.get('time_series_score'):
					n_splits = param_dict['time_series_score'].get('n_splits',5)
					modelValidator(self.gnbTrained_i,trainF_i,trainR_i).time_series_validation(n_splits)

			self.testIDNum -= 1
			

	@timeLog('cyclicalTrailingMethod','total')
	def adaboost(self,mission='classification',*cvargs,**kwargs):
		'''
		*cvargs: validation in/out sample
		default :cvNum=5,testSize=0.3
		**kwargs: args dict to model  
		'''
		testTimeIDL,timeIDUQL,timeIDUQ = self.timeIndexInitial()
		generator = self.TrTeGenerator(testTimeIDL,timeIDUQL,timeIDUQ)
		runningMetric_ = runningMetric(self.testIDNum)
		while self.testIDNum > 0: 
			trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i,modelfilName_i,datafilName_i,\
				testMetricF_i,testMetricR_i,testStockID_i = next(generator)		#generator.next() for python 2.x
			self.adaTrained_i,self.adaTestScore_i = \
				modelTrainer(self.paraTuneMethod,trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i)\
					.adaboost(mission=mission,**kwargs)
			print('%s_%s_%s testScore:%s'%(self.__trainMethod,'adaboost',self.paraTuneMethod,self.adaTestScore_i))
			runningMetric_.modelRankIC(mission,testF_i,testR_i,self.adaTrained_i)
			fileAccess().modelStorer(self.adaTrained_i,self.__trainMethod,'adaboost',self.paraTuneMethod,self.ortho_method,modelfilName_i)


			if len(cvargs[0]) != 0:
				param_dict = cvargs[0]

				if param_dict.get('cross_val_score'):
					cv = param_dict['cross_val_score'].get('cv',4)

					if mission == 'classification':
						scoring = param_dict['cross_val_score'].get('cross_val_score','roc_auc')
						modelValidator(self.adaTrained_i,trainF_i,trainL_i).cross_validation(cv,scoring)
					elif mission == 'regression':
						scoring = param_dict['cross_val_score'].get('cross_val_score','neg_mean_squared_error')
						modelValidator(self.adaTrained_i,trainF_i,trainR_i).cross_validation(cv,scoring)
					
				elif param_dict.get('time_series_score'):
					n_splits = param_dict['time_series_score'].get('n_splits',5)

					if mission == 'classification':
						modelValidator(self.adaTrained_i,trainF_i,trainL_i).time_series_validation(n_splits)
					elif mission == 'regression':
						modelValidator(self.adaTrained_i,trainF_i,trainR_i).time_series_validation(n_splits)
			
			self.testIDNum -= 1
			

	@timeLog('cyclicalTrailingMethod','total')
	def xgboost(self,mission='classification',*cvargs,**kwargs):
		testTimeIDL,timeIDUQL,timeIDUQ = self.timeIndexInitial()
		generator = self.TrTeGenerator(testTimeIDL,timeIDUQL,timeIDUQ)
		runningMetric_ = runningMetric(self.testIDNum)
		while self.testIDNum > 0: 
			trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i,modelfilName_i,datafilName_i,\
				testMetricF_i,testMetricR_i,testStockID_i = next(generator)		#generator.next() for python 2.x
			self.xgbTrained_i,self.xgbTestScore_i = \
				modelTrainer(self.paraTuneMethod,trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i)\
					.xgboost(mission=mission,**kwargs)
			print('%s_%s_%s testScore:%s'%(self.__trainMethod,'xgboost',self.paraTuneMethod,self.xgbTestScore_i))
			runningMetric_.modelRankIC(mission,testF_i,testR_i,self.xgbTrained_i)
			fileAccess().modelStorer(self.xgbTrained_i,self.__trainMethod,'xgboost',self.paraTuneMethod,self.ortho_method,modelfilName_i)


			if len(cvargs[0]) != 0:
				param_dict = cvargs[0]

				if param_dict.get('cross_val_score'):
					cv = param_dict['cross_val_score'].get('cv',4)

					if mission == 'classification':
						scoring = param_dict['cross_val_score'].get('cross_val_score','roc_auc')
						modelValidator(self.xgbTrained_i,trainF_i,trainL_i).cross_validation(cv,scoring)
					elif mission == 'regression':
						scoring = param_dict['cross_val_score'].get('cross_val_score','neg_mean_squared_error')
						modelValidator(self.xgbTrained_i,trainF_i,trainR_i).cross_validation(cv,scoring)
					
				elif param_dict.get('time_series_score'):
					n_splits = param_dict['time_series_score'].get('n_splits',5)

					if mission == 'classification':
						modelValidator(self.xgbTrained_i,trainF_i,trainL_i).time_series_validation(n_splits)
					elif mission == 'regression':
						modelValidator(self.xgbTrained_i,trainF_i,trainR_i).time_series_validation(n_splits)

			self.testIDNum -= 1
			

	@timeLog('cyclicalTrailingMethod','total')
	def lightgbm(self,mission='classification',*cvargs,**kwargs):
		testTimeIDL,timeIDUQL,timeIDUQ = self.timeIndexInitial()
		generator = self.TrTeGenerator(testTimeIDL,timeIDUQL,timeIDUQ)
		runningMetric_ = runningMetric(self.testIDNum)
		while self.testIDNum > 0: 
			trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i,modelfilName_i,datafilName_i,\
				testMetricF_i,testMetricR_i,testStockID_i = next(generator)		#generator.next() for python 2.x
			self.lgbTrained_i,self.lgbTestScore_i = \
				modelTrainer(self.paraTuneMethod,trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i)\
					.lightgbm(mission=mission,**kwargs)
			print('%s_%s_%s testScore:%s'%(self.__trainMethod,'lightgbm',self.paraTuneMethod,self.lgbTestScore_i))
			runningMetric_.modelRankIC(mission,testF_i,testR_i,self.lgbTrained_i)
			fileAccess().modelStorer(self.lgbTrained_i,self.__trainMethod,'lightgbm',self.paraTuneMethod,self.ortho_method,modelfilName_i)


			if len(cvargs[0]) != 0:
				param_dict = cvargs[0]

				if param_dict.get('cross_val_score'):
					cv = param_dict['cross_val_score'].get('cv',4)

					if mission == 'classification':
						scoring = param_dict['cross_val_score'].get('cross_val_score','roc_auc')
						modelValidator(self.lgbTrained_i,trainF_i,trainL_i).cross_validation(cv,scoring)
					elif mission == 'regression':
						scoring = param_dict['cross_val_score'].get('cross_val_score','neg_mean_squared_error')
						modelValidator(self.lgbTrained_i,trainF_i,trainR_i).cross_validation(cv,scoring)
					
				elif param_dict.get('time_series_score'):
					n_splits = param_dict['time_series_score'].get('n_splits',5)

					if mission == 'classification':
						modelValidator(self.lgbTrained_i,trainF_i,trainL_i).time_series_validation(n_splits)
					elif mission == 'regression':
						modelValidator(self.lgbTrained_i,trainF_i,trainR_i).time_series_validation(n_splits)
			
			self.testIDNum -= 1
			
	
	@timeLog('cyclicalTrailingMethod','total')
	def catboost(self,mission='classification',*cvargs,**kwargs):
		testTimeIDL,timeIDUQL,timeIDUQ = self.timeIndexInitial()
		generator = self.TrTeGenerator(testTimeIDL,timeIDUQL,timeIDUQ)
		runningMetric_ = runningMetric(self.testIDNum)
		while self.testIDNum > 0: 
			trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i,modelfilName_i,datafilName_i,\
				testMetricF_i,testMetricR_i,testStockID_i = next(generator)		#generator.next() for python 2.x
			self.cbTrained_i,self.cbTestScore_i = \
				modelTrainer(self.paraTuneMethod,trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i)\
					.catboost(mission=mission,**kwargs)
			print('%s_%s_%s testScore:%s'%(self.__trainMethod,'catboost',self.paraTuneMethod,self.cbTestScore_i))
			runningMetric_.modelRankIC(mission,testF_i,testR_i,self.cbTrained_i)
			fileAccess().modelStorer(self.cbTrained_i,self.__trainMethod,'catboost',self.paraTuneMethod,self.ortho_method,modelfilName_i)

			
			if len(cvargs[0]) != 0:
				param_dict = cvargs[0]

				if param_dict.get('cross_val_score'):
					cv = param_dict['cross_val_score'].get('cv',4)

					if mission == 'classification':
						scoring = param_dict['cross_val_score'].get('cross_val_score','roc_auc')
						modelValidator(self.cbTrained_i,trainF_i,trainL_i).cross_validation(cv,scoring)
					elif mission == 'regression':
						scoring = param_dict['cross_val_score'].get('cross_val_score','neg_mean_squared_error')
						modelValidator(self.cbTrained_i,trainF_i,trainR_i).cross_validation(cv,scoring)
					
				elif param_dict.get('time_series_score'):
					n_splits = param_dict['time_series_score'].get('n_splits',5)

					if mission == 'classification':
						modelValidator(self.cbTrained_i,trainF_i,trainL_i).time_series_validation(n_splits)
					elif mission == 'regression':
						modelValidator(self.cbTrained_i,trainF_i,trainR_i).time_series_validation(n_splits)

			self.testIDNum -= 1
			


	@timeLog('cyclicalTrailingMethod','total')
	def svm(self,mission='classification',*cvargs,**kwargs):
		testTimeIDL,timeIDUQL,timeIDUQ = self.timeIndexInitial()
		generator = self.TrTeGenerator(testTimeIDL,timeIDUQL,timeIDUQ)
		runningMetric_ = runningMetric(self.testIDNum)
		while self.testIDNum > 0: 
			trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i,modelfilName_i,datafilName_i,\
				testMetricF_i,testMetricR_i,testStockID_i = next(generator)		#generator.next() for python 2.x
			self.svmTrained_i,self.svmTestScore_i = \
				modelTrainer(self.paraTuneMethod,trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i)\
					.svm(mission=mission,**kwargs)
			print('%s_%s_%s testScore:%s'%(self.__trainMethod,'svm',self.paraTuneMethod,self.svmTestScore_i))
			runningMetric_.modelRankIC(mission,testF_i,testR_i,self.svmTrained_i)	
			fileAccess().modelStorer(self.svmTrained_i,self.__trainMethod,'svm',self.paraTuneMethod,self.ortho_method,modelfilName_i)

			
			if len(cvargs[0]) != 0:
				param_dict = cvargs[0]

				if param_dict.get('cross_val_score'):
					cv = param_dict['cross_val_score'].get('cv',4)
					scoring = param_dict['cross_val_score'].get('cross_val_score','roc_auc')
					modelValidator(self.svmTrained_i,trainF_i,trainR_i).cross_validation(cv,scoring)

				elif param_dict.get('time_series_score'):
					n_splits = param_dict['time_series_score'].get('n_splits',5)
					modelValidator(self.svmTrained_i,trainF_i,trainR_i).time_series_validation(n_splits)

			self.testIDNum -= 1
			

	@timeLog('cyclicalTrailingMethod','total')
	def logistic(self,mission='classification',*cvargs,**kwargs):
		testTimeIDL,timeIDUQL,timeIDUQ = self.timeIndexInitial()
		generator = self.TrTeGenerator(testTimeIDL,timeIDUQL,timeIDUQ)
		runningMetric_ = runningMetric(self.testIDNum)
		while self.testIDNum > 0: 
			trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i,modelfilName_i,datafilName_i,\
				testMetricF_i,testMetricR_i,testStockID_i = next(generator)		#generator.next() for python 2.x
			self.logisticTrained_i,self.logisticTestScore_i = \
				modelTrainer(self.paraTuneMethod,trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i)\
					.logistic(mission=mission,**kwargs)
			print('%s_%s_%s testScore:%s'%(self.__trainMethod,'logistic',self.paraTuneMethod,self.logisticTestScore_i))
			runningMetric_.modelRankIC(mission,testF_i,testR_i,self.logisticTrained_i)	
			fileAccess().modelStorer(self.logisticTrained_i,self.__trainMethod,'logistic',self.paraTuneMethod,self.ortho_method,modelfilName_i)

			
			if len(cvargs[0]) != 0:
				param_dict = cvargs[0]

				if param_dict.get('cross_val_score'):
					cv = param_dict['cross_val_score'].get('cv',4)
					scoring = param_dict['cross_val_score'].get('cross_val_score','roc_auc')
					modelValidator(self.logisticTrained_i,trainF_i,trainR_i).cross_validation(cv,scoring)

				elif param_dict.get('time_series_score'):
					n_splits = param_dict['time_series_score'].get('n_splits',5)
					modelValidator(self.logisticTrained_i,trainF_i,trainR_i).time_series_validation(n_splits)

			self.testIDNum -= 1


	@timeLog('cyclicalTrailingMethod','total')
	def linearSVC(self,mission='classification',*cvargs,**kwargs):
		testTimeIDL,timeIDUQL,timeIDUQ = self.timeIndexInitial()
		generator = self.TrTeGenerator(testTimeIDL,timeIDUQL,timeIDUQ)
		runningMetric_ = runningMetric(self.testIDNum)
		while self.testIDNum > 0: 
			trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i,modelfilName_i,datafilName_i,\
				testMetricF_i,testMetricR_i,testStockID_i = next(generator)		#generator.next() for python 2.x
			self.linearSVCTrained_i,self.linearSVCTestScore_i = \
				modelTrainer(self.paraTuneMethod,trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i)\
					.linearSVC(mission=mission,**kwargs)
			print('%s_%s_%s testScore:%s'%(self.__trainMethod,'linearsvc',self.paraTuneMethod,self.linearSVCTestScore_i))
			runningMetric_.modelRankIC(mission,testF_i,testR_i,self.linearSVCTrained_i)	
			fileAccess().modelStorer(self.linearSVCTrained_i,self.__trainMethod,'linearsvc',self.paraTuneMethod,self.ortho_method,modelfilName_i)

			
			if len(cvargs[0]) != 0:
				param_dict = cvargs[0]

				if param_dict.get('cross_val_score'):
					cv = param_dict['cross_val_score'].get('cv',4)
					scoring = param_dict['cross_val_score'].get('cross_val_score','roc_auc')
					modelValidator(self.linearSVCTrained_i,trainF_i,trainR_i).cross_validation(cv,scoring)

				elif param_dict.get('time_series_score'):
					n_splits = param_dict['time_series_score'].get('n_splits',5)
					modelValidator(self.linearSVCTrained_i,trainF_i,trainR_i).time_series_validation(n_splits)

			self.testIDNum -= 1
			

	@timeLog('cyclicalTrailingMethod','total')
	def sgd(self,mission='classification',*cvargs,**kwargs):
		testTimeIDL,timeIDUQL,timeIDUQ = self.timeIndexInitial()
		generator = self.TrTeGenerator(testTimeIDL,timeIDUQL,timeIDUQ)
		runningMetric_ = runningMetric(self.testIDNum)
		while self.testIDNum > 0: 
			trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i,modelfilName_i,datafilName_i,\
				testMetricF_i,testMetricR_i,testStockID_i = next(generator)		#generator.next() for python 2.x
			self.sgdTrained_i,self.sgdTestScore_i = \
				modelTrainer(self.paraTuneMethod,trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i)\
					.sgd(mission=mission,**kwargs)
			print('%s_%s_%s testScore:%s'%(self.__trainMethod,'sgd',self.paraTuneMethod,self.sgdTestScore_i))
			runningMetric_.modelRankIC(mission,testF_i,testR_i,self.sgdTrained_i)	
			fileAccess().modelStorer(self.sgdTrained_i,self.__trainMethod,'sgd',self.paraTuneMethod,self.ortho_method,modelfilName_i)

			
			if len(cvargs[0]) != 0:
				param_dict = cvargs[0]

				if param_dict.get('cross_val_score'):
					cv = param_dict['cross_val_score'].get('cv',4)

					if mission == 'classification':
						scoring = param_dict['cross_val_score'].get('cross_val_score','roc_auc')
						modelValidator(self.sgdTrained_i,trainF_i,trainL_i).cross_validation(cv,scoring)
					elif mission == 'regression':
						scoring = param_dict['cross_val_score'].get('cross_val_score','neg_mean_squared_error')
						modelValidator(self.sgdTrained_i,trainF_i,trainR_i).cross_validation(cv,scoring)
					
				elif param_dict.get('time_series_score'):
					n_splits = param_dict['time_series_score'].get('n_splits',5)

					if mission == 'classification':
						modelValidator(self.sgdTrained_i,trainF_i,trainL_i).time_series_validation(n_splits)
					elif mission == 'regression':
						modelValidator(self.sgdTrained_i,trainF_i,trainR_i).time_series_validation(n_splits)

			self.testIDNum -= 1
				

	@timeLog('cyclicalTrailingMethod','total')
	def deepforest(self,mission='classification',*cvargs,**kwargs):
		testTimeIDL,timeIDUQL,timeIDUQ = self.timeIndexInitial()
		generator = self.TrTeGenerator(testTimeIDL,timeIDUQL,timeIDUQ)
		runningMetric_ = runningMetric(self.testIDNum)
		while self.testIDNum > 0: 
			trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i,modelfilName_i,datafilName_i,\
				testMetricF_i,testMetricR_i,testStockID_i = next(generator)		#generator.next() for python 2.x
			self.dfTrained_i,self.dfTestScore_i = \
				modelTrainer(self.paraTuneMethod,trainF_i,trainL_i,trainR_i,testF_i,testL_i,testR_i,trainTimeID_i,trainStockID_i)\
					.deepforest(mission=mission,**kwargs)
			print('%s_%s_%s testScore:%s'%(self.__trainMethod,'deepforest',self.paraTuneMethod,self.dfTestScore_i))
			runningMetric_.modelRankIC(mission,testF_i,testR_i,self.dfTrained_i)	
			fileAccess().modelStorer(self.dfTrained_i,self.__trainMethod,'deepforest',self.paraTuneMethod,self.ortho_method,modelfilName_i)

			
			if len(cvargs[0]) != 0:
				param_dict = cvargs[0]

				if param_dict.get('cross_val_score'):
					cv = param_dict['cross_val_score'].get('cv',4)

					if mission == 'classification':
						scoring = param_dict['cross_val_score'].get('cross_val_score','roc_auc')
						modelValidator(self.dfTrained_i,trainF_i,trainL_i).cross_validation(cv,scoring)
					elif mission == 'regression':
						scoring = param_dict['cross_val_score'].get('cross_val_score','neg_mean_squared_error')
						modelValidator(self.dfTrained_i,trainF_i,trainR_i).cross_validation(cv,scoring)
					
				elif param_dict.get('time_series_score'):
					n_splits = param_dict['time_series_score'].get('n_splits',5)

					if mission == 'classification':
						modelValidator(self.dfTrained_i,trainF_i,trainL_i).time_series_validation(n_splits)
					elif mission == 'regression':
						modelValidator(self.dfTrained_i,trainF_i,trainR_i).time_series_validation(n_splits)

			self.testIDNum -= 1