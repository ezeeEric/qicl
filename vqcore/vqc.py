# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import logging
import math
import time
import numpy as np

from sklearn.utils import shuffle
from sklearn.metrics import log_loss

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.aqua import Pluggable, PluggableType, get_pluggable_class, AquaError
from qiskit.aqua.utils import get_feature_dimension
from qiskit.aqua.utils import map_label_to_class_name
from qiskit.aqua.utils import split_dataset_to_data_and_labels
from qiskit.aqua.algorithms.adaptive.vq_algorithm import VQAlgorithm

from utils.tools import timelogDict,accumulativeChronomat,accumulativeChronomat

from vqcore.vqc_utils import assign_label,cost_estimate,cost_estimate_sigmoid,return_probabilities

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class VQC(VQAlgorithm):

    CONFIGURATION = {
        'name': 'VQC',
        'description': 'Variational Quantum Classifier',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'vqc_schema',
            'type': 'object',
            'properties': {
                'override_SPSA_params': {
                    'type': 'boolean',
                    'default': True
                },
                'max_evals_grouped': {
                    'type': 'integer',
                    'default': 1
                },
                'minibatch_size': {
                    'type': 'integer',
                    'default': -1
                }
            },
            'additionalProperties': False
        },
        'problems': ['classification'],
        'depends': [
            {
                'pluggable_type': 'optimizer',
                'default': {
                    'name': 'SPSA'
                },
            },
            {
                'pluggable_type': 'feature_map',
                'default': {
                    'name': 'SecondOrderExpansion',
                    'depth': 2
                },
            },
            {
                'pluggable_type': 'variational_form',
                'default': {
                    'name': 'RYRZ',
                    'depth': 3
                },
            },
        ],
    }

    def __init__(
            self,
            optimizer=None,
            feature_map=None,
            var_form=None,
            training_dataset=None,
            test_dataset=None,
            datapoints=None,
            max_evals_grouped=1,
            minibatch_size=-1,
            callback=None,

    ):
        """Initialize the object
        Args:
            optimizer (Optimizer): The classical optimizer to use.
            feature_map (FeatureMap): The FeatureMap instance to use.
            var_form (VariationalForm): The variational form instance.
            training_dataset (dict): The training dataset, in the format: {'A': np.ndarray, 'B': np.ndarray, ...}.
            test_dataset (dict): The test dataset, in same format as `training_dataset`.
            datapoints (np.ndarray): NxD array, N is the number of data and D is data dimension.
            max_evals_grouped (int): The maximum number of evaluations to perform simultaneously.
            minibatch_size (int): The size of a mini-batch.
            callback (Callable): a callback that can access the intermediate data during the optimization.
                Internally, four arguments are provided as follows the index of data batch, the index of evaluation,
                parameters of variational form, evaluated value.
        Notes:
            We use `label` to denotes numeric results and `class` the class names (str).
        """

        self.validate(locals())
        super().__init__(
            var_form=var_form,
            optimizer=optimizer,
            cost_fn=self._cost_function_wrapper
        )
        self._optimizer.set_max_evals_grouped(max_evals_grouped)

        self._callback = callback

        if feature_map is None:
            raise AquaError('Missing feature map.')
        if training_dataset is None:
            raise AquaError('Missing training dataset.')
        self._training_dataset, self._class_to_label = split_dataset_to_data_and_labels(
            training_dataset)
        self._label_to_class = {label: class_name for class_name, label
                                in self._class_to_label.items()}
        self._num_classes = len(list(self._class_to_label.keys()))

        if test_dataset is not None:
            self._test_dataset = split_dataset_to_data_and_labels(test_dataset,
                                                                  self._class_to_label)
        else:
            self._test_dataset = test_dataset

        if datapoints is not None and not isinstance(datapoints, np.ndarray):
            datapoints = np.asarray(datapoints)
            if len(datapoints) == 0:
                datapoints = None
        self._datapoints = datapoints
        self._minibatch_size = minibatch_size

        self._eval_count = 0
        self._ret = {}
        self._feature_map = feature_map
        self._num_qubits = feature_map.num_qubits
        
        self.qcl_maxiter=1
        self.qcl_n_iter=1
        self.qcl_y_list=self._training_dataset[1]

        self.itCounter={}

    @classmethod
    def init_params(cls, params, algo_input):
        algo_params = params.get(Pluggable.SECTION_KEY_ALGORITHM)
        override_spsa_params = algo_params.get('override_SPSA_params')
        max_evals_grouped = algo_params.get('max_evals_grouped')
        minibatch_size = algo_params.get('minibatch_size')

        # Set up optimizer
        opt_params = params.get(Pluggable.SECTION_KEY_OPTIMIZER)
        # If SPSA then override SPSA params as reqd to our predetermined values
        if opt_params['name'] == 'SPSA' and override_spsa_params:
            opt_params['c0'] = 4.0
            opt_params['c1'] = 0.1
            opt_params['c2'] = 0.602
            opt_params['c3'] = 0.101
            opt_params['c4'] = 0.0
            opt_params['skip_calibration'] = True
        optimizer = get_pluggable_class(PluggableType.OPTIMIZER,
                                        opt_params['name']).init_params(params)

        # Set up feature map
        fea_map_params = params.get(Pluggable.SECTION_KEY_FEATURE_MAP)
        feature_dimension = get_feature_dimension(algo_input.training_dataset)
        fea_map_params['feature_dimension'] = feature_dimension
        feature_map = get_pluggable_class(PluggableType.FEATURE_MAP,
                                          fea_map_params['name']).init_params(params)

        # Set up variational form, we need to add computed num qubits
        # Pass all parameters so that Variational Form can create its dependents
        var_form_params = params.get(Pluggable.SECTION_KEY_VAR_FORM)
        var_form_params['num_qubits'] = feature_map.num_qubits
        var_form = get_pluggable_class(PluggableType.VARIATIONAL_FORM,
                                       var_form_params['name']).init_params(params)

        return cls(optimizer, feature_map, var_form, algo_input.training_dataset,
                   algo_input.test_dataset, algo_input.datapoints, max_evals_grouped,
                   minibatch_size)
    
    @accumulativeChronomat
    def construct_circuit(self, x, theta, measurement=False):
        """
        Construct circuit based on data and parameters in variational form.

        Args:
            x (numpy.ndarray): 1-D array with D dimension
            theta ([numpy.ndarray]): list of 1-D array, parameters sets for variational form
            measurement (bool): flag to add measurement
        Returns:
            QuantumCircuit: the circuit
        """
        qr = QuantumRegister(self._num_qubits, name='q')
        cr = ClassicalRegister(self._num_qubits, name='c')
        qc = QuantumCircuit(qr, cr)
        qc += self.timedFMconstruct_circuit(x, qr)
        qc += self.timedVFconstruct_circuit(theta, qr)
        if measurement:
            #"fixes" circuit block
            #any compilerside modifications contained within that block.
            qc.barrier(qr)
            self.timed_qc_measure(qc, qr, cr)
            #qc.measure(qr, cr)
        return qc
    
    @accumulativeChronomat
    def timed_qc_measure(self, qc, qr, cr):
        qc.measure(qr, cr)

    @accumulativeChronomat
    def timedFMconstruct_circuit(self, x,qr):
        return self._feature_map.construct_circuit(x, qr)
    
    @accumulativeChronomat
    def timedVFconstruct_circuit(self,theta,qr):
        return self._var_form.construct_circuit(theta,qr)
    
    @accumulativeChronomat
    def timedQIexecute(self,vals):
        return self._quantum_instance.execute(vals)
     
    @accumulativeChronomat
    def _get_prediction(self, data, theta):
        """
        Make prediction on data based on each theta.

        Args:
            data (numpy.ndarray): 2-D array, NxD, N data points, each with D dimension
            theta ([numpy.ndarray]): list of 1-D array, parameters sets for variational form
        Returns:
            numpy.ndarray or [numpy.ndarray]: list of NxK array
            numpy.ndarray or [numpy.ndarray]: list of Nx1 array
        """
        # if self._quantum_instance.is_statevector:
        #     raise ValueError('Selected backend "{}" is not supported.'.format(
        #         self._quantum_instance.backend_name))

        circuits = {}
        circuit_id = 0

        num_theta_sets = len(theta) // self._var_form.num_parameters
        theta_sets = np.split(theta, num_theta_sets)

        for theta in theta_sets:
            for datum in data:
                if self._quantum_instance.is_statevector:
                    circuit = self.construct_circuit(datum, theta, measurement=False)
                else:
                    circuit = self.construct_circuit(datum, theta, measurement=True)

                circuits[circuit_id] = circuit
                circuit_id += 1

        #results = self._quantum_instance.execute(list(circuits.values()))
        results = self.timedQIexecute(list(circuits.values()))

        circuit_id = 0
        predicted_probs = []
        predicted_labels = []
        for _ in theta_sets:
            counts = []
            for _ in data:
                if self._quantum_instance.is_statevector:
                    temp = results.get_statevector(circuits[circuit_id])
                    outcome_vector = (temp * temp.conj()).real
                    # convert outcome_vector to outcome_dict, where key is a basis state and value is the count.
                    # Note: the count can be scaled linearly, i.e., it does not have to be an integer.
                    outcome_dict = {}
                    bitstr_size = int(math.log2(len(outcome_vector)))
                    for i in range(len(outcome_vector)):
                        bitstr_i = format(i, '0' + str(bitstr_size) + 'b')
                        outcome_dict[bitstr_i] = outcome_vector[i]
                else:
                    outcome_dict = results.get_counts(circuits[circuit_id])

                counts.append(outcome_dict)
                circuit_id += 1

            probs = return_probabilities(counts, self._num_classes)
            predicted_probs.append(probs)
            predicted_labels.append(np.argmax(probs, axis=1))

        if len(predicted_probs) == 1:
            predicted_probs = predicted_probs[0]
        if len(predicted_labels) == 1:
            predicted_labels = predicted_labels[0]

        return predicted_probs, predicted_labels

    # Breaks data into minibatches. Labels are optional, but will be broken into batches if included.
    def batch_data(self, data, labels=None, minibatch_size=-1):
        label_batches = None

        if 0 < minibatch_size < len(data):
            batch_size = min(minibatch_size, len(data))
            if labels is not None:
                shuffled_samples, shuffled_labels = shuffle(data, labels, random_state=self.random)
                label_batches = np.array_split(shuffled_labels, batch_size)
            else:
                shuffled_samples = shuffle(data, random_state=self.random)
            batches = np.array_split(shuffled_samples, batch_size)
        else:
            batches = np.asarray([data])
            label_batches = np.asarray([labels])
        return batches, label_batches

    def is_gradient_really_supported(self):
        return self.optimizer.is_gradient_supported and not self.optimizer.is_gradient_ignored
  
    @accumulativeChronomat
    def train(self, data, labels, quantum_instance=None, minibatch_size=-1):
        """Train the models, and save results.

        Args:
            data (numpy.ndarray): NxD array, N is number of data and D is dimension
            labels (numpy.ndarray): Nx1 array, N is number of data
            quantum_instance (QuantumInstance): quantum backend with all setting
            minibatch_size (int): the size of each minibatched accuracy evalutation
        """
        self._quantum_instance = self._quantum_instance if quantum_instance is None else quantum_instance
        minibatch_size = minibatch_size if minibatch_size > 0 else self._minibatch_size
        self._batches, self._label_batches = self.batch_data(data, labels, minibatch_size)
        self._batch_index = 0

        if self.initial_point is None:
            self.initial_point = self.random.randn(self._var_form.num_parameters)

        self._eval_count = 0

        grad_fn = None
        if minibatch_size > 0 and self.is_gradient_really_supported():  # we need some wrapper
            grad_fn = self._gradient_function_wrapper

        self._ret = self.find_minimum(
            initial_point=self.initial_point,
            var_form=self.var_form,
            cost_fn=self._cost_function_wrapper,
            optimizer=self.optimizer,
            gradient_fn=grad_fn  # func for computing gradient
        )

        if self._ret['num_optimizer_evals'] is not None and self._eval_count >= self._ret['num_optimizer_evals']:
            self._eval_count = self._ret['num_optimizer_evals']
        self._eval_time = self._ret['eval_time']
        logger.info('Optimization complete in {} seconds.\nFound opt_params {} in {} evals'.format(
            self._eval_time, self._ret['opt_params'], self._eval_count))
        self._ret['eval_count'] = self._eval_count

        del self._batches
        del self._label_batches
        del self._batch_index

        self._ret['training_loss'] = self._ret['min_val']
    
    
    def qcl_pred(self,theta):
        predicted_probs, predicted_labels = self._get_prediction(self._training_dataset[0], theta)
        return predicted_labels

    def qcl_cost_func(self, theta):
        """QCL adapted Cost function
        :param theta: List of rotation angle theta
        """
        y_pred = self.qcl_pred(theta)
        # cross-entropy loss
        loss = log_loss(self.qcl_y_list, y_pred)
        return loss
    
    # for BFGS
    def qcl_b_grad(self, theta):
        # Return the list of dB/dtheta
        theta_plus = [theta.copy() + np.eye(len(theta))[i] * np.pi / 2. for i in range(len(theta))]
        theta_minus = [theta.copy() - np.eye(len(theta))[i] * np.pi / 2. for i in range(len(theta))]

        grad = [(self.qcl_pred(theta_plus[i]) - self.qcl_pred(theta_minus[i])) / 2. for i in range(len(theta))]

        return np.array(grad)

    # for BFGS
    def qcl_cost_func_grad(self, theta):
        y_minus_t = self.qcl_pred(theta) - self.qcl_y_list
        B_gr_list = self.qcl_b_grad(theta)
        grad = [np.sum(y_minus_t * B_gr) for B_gr in B_gr_list]
        return np.array(grad)
    
    def qcl_callbackF(self, theta):
            self.qcl_n_iter = self.qcl_n_iter + 1
            if 10 * self.qcl_n_iter % self.qcl_maxiter == 0:
                print("Iteration: ",self.qcl_n_iter / self.qcl_maxiter,",  Value of cost_func: ",self.qcl_cost_func(theta))
    
    @accumulativeChronomat
    def find_minimum(self, initial_point=None, var_form=None, cost_fn=None, optimizer=None, gradient_fn=None):
        """Optimize to find the minimum cost value.
           #OVERWRITTEN
        Returns:
            Optimized variational parameters, and corresponding minimum cost value.

        Raises:
            ValueError:

        """
        initial_point = initial_point if initial_point is not None else self._initial_point
        var_form = var_form if var_form is not None else self._var_form
        cost_fn = cost_fn if cost_fn is not None else self._cost_fn
        optimizer = optimizer if optimizer is not None else self._optimizer

        nparms = var_form.num_parameters
        bounds = var_form.parameter_bounds

        if initial_point is not None and len(initial_point) != nparms:
            raise ValueError('Initial point size {} and parameter size {} mismatch'.format(len(initial_point), nparms))
        if len(bounds) != nparms:
            raise ValueError('Variational form bounds size does not match parameter size')
        # If *any* value is *equal* in bounds array to None then the problem does *not* have bounds
        problem_has_bounds = not np.any(np.equal(bounds, None))
        # Check capabilities of the optimizer
        if problem_has_bounds:
            if not optimizer.is_bounds_supported:
                raise ValueError('Problem has bounds but optimizer does not support bounds')
        else:
            if optimizer.is_bounds_required:
                raise ValueError('Problem does not have bounds but optimizer requires bounds')
        if initial_point is not None:
            if not optimizer.is_initial_point_supported:
                raise ValueError('Optimizer does not support initial point')
        else:
            if optimizer.is_initial_point_required:
                low = [(l if l is not None else -2 * np.pi) for (l, u) in bounds]
                high = [(u if u is not None else 2 * np.pi) for (l, u) in bounds]
                initial_point = self.random.uniform(low, high)

        start = time.time()
        if not optimizer.is_gradient_supported:  # ignore the passed gradient function
            gradient_fn = None

        logger.info('Starting optimizer.\nbounds={}\ninitial point={}'.format(bounds, initial_point))
        
        ret = {}
        opt_params, opt_val, num_optimizer_evals = optimizer.optimize(var_form.num_parameters,
                                                                      cost_fn,
                                                                      variable_bounds=bounds,
                                                                      initial_point=initial_point,
                                                                      gradient_function=gradient_fn)
        ret['num_optimizer_evals'] = num_optimizer_evals
        ret['min_val'] = opt_val
        ret['opt_params'] = opt_params

        eval_time = time.time() - start
        ret['eval_time'] = eval_time

        return ret

    # temporary fix: this code should be unified with the gradient api in optimizer.py
    def _gradient_function_wrapper(self, theta):
        """Compute and return the gradient at the point theta.
        Args:
            theta (numpy.ndarray): 1-d array
        Returns:
            numpy.ndarray: 1-d array with the same shape as theta. The  gradient computed
        """
        epsilon = 1e-8
        f_orig = self._cost_function_wrapper(theta)
        grad = np.zeros((len(theta),), float)
        for k in range(len(theta)):
            theta[k] += epsilon
            f_new = self._cost_function_wrapper(theta)
            grad[k] = (f_new - f_orig) / epsilon
            theta[k] -= epsilon  # recover to the center state
        if self.is_gradient_really_supported():
            self._batch_index += 1  # increment the batch after gradient callback
        return grad
    
    @accumulativeChronomat
    def _cost_function_wrapper(self, theta):
        batch_index = self._batch_index % len(self._batches)
        if batch_index in self.itCounter.keys():
            self.itCounter[batch_index]+=1
        else:
            self.itCounter[batch_index]=1
        if batch_index%5==0:
            print("Optimiser iteration {0} - batch {1}".format(self.itCounter[batch_index],batch_index))
        predicted_probs, predicted_labels = self._get_prediction(self._batches[batch_index], theta)
        total_cost = []
        if not isinstance(predicted_probs, list):
            predicted_probs = [predicted_probs]
        for i in range(len(predicted_probs)):
            curr_cost = cost_estimate(predicted_probs[i], self._label_batches[batch_index])
            total_cost.append(curr_cost)
            if self._callback is not None:
                self._callback(
                    self._eval_count,
                    theta[i * self._var_form.num_parameters:(i + 1) * self._var_form.num_parameters],
                    curr_cost,
                    self._batch_index
                )
            self._eval_count += 1
        if not self.is_gradient_really_supported():
            self._batch_index += 1  # increment the batch after eval callback

        logger.debug('Intermediate batch cost: {}'.format(sum(total_cost)))
        return total_cost if len(total_cost) > 1 else total_cost[0]
  
    @accumulativeChronomat
    def test(self, data, labels, quantum_instance=None, minibatch_size=-1, params=None):
        """Predict the labels for the data, and test against with ground truth labels.

        Args:
            data (numpy.ndarray): NxD array, N is number of data and D is data dimension
            labels (numpy.ndarray): Nx1 array, N is number of data
            quantum_instance (QuantumInstance): quantum backend with all setting
            minibatch_size (int): the size of each minibatched accuracy evalutation
            params (list): list of parameters to populate in the variational form
        Returns:
            float: classification accuracy
        """
        # minibatch size defaults to setting in instance variable if not set
        minibatch_size = minibatch_size if minibatch_size > 0 else self._minibatch_size

        batches, label_batches = self.batch_data(data, labels, minibatch_size)
        self.batch_num = 0
        if params is None:
            params = self.optimal_params
        total_cost = 0
        total_correct = 0
        total_samples = 0

        self._quantum_instance = self._quantum_instance if quantum_instance is None else quantum_instance
        for batch, label_batch in zip(batches, label_batches):
            predicted_probs, predicted_labels = self._get_prediction(batch, params)
            total_cost += cost_estimate(predicted_probs, label_batch)
            total_correct += np.sum((np.argmax(predicted_probs, axis=1) == label_batch))
            total_samples += label_batch.shape[0]
            int_accuracy = np.sum((np.argmax(predicted_probs, axis=1) == label_batch)) / label_batch.shape[0]
            logger.debug('Intermediate batch accuracy: {:.2f}%'.format(int_accuracy * 100.0))
        total_accuracy = total_correct / total_samples
        logger.info('Accuracy is {:.2f}%'.format(total_accuracy * 100.0))
        self._ret['testing_accuracy'] = total_accuracy
        self._ret['test_success_ratio'] = total_accuracy
        self._ret['testing_loss'] = total_cost / len(batches)
        return total_accuracy

    @accumulativeChronomat
    def predict(self, data, quantum_instance=None, minibatch_size=-1, params=None):
        """Predict the labels for the data.

        Args:
            data (numpy.ndarray): NxD array, N is number of data, D is data dimension
            quantum_instance (QuantumInstance): quantum backend with all setting
            minibatch_size (int): the size of each minibatched accuracy evalutation
            params (list): list of parameters to populate in the variational form
        Returns:
            list: for each data point, generates the predicted probability for each class
            list: for each data point, generates the predicted label (that with the highest prob)
        """

        # minibatch size defaults to setting in instance variable if not set
        minibatch_size = minibatch_size if minibatch_size > 0 else self._minibatch_size
        batches, _ = self.batch_data(data, None, minibatch_size)
        if params is None:
            params = self.optimal_params
        predicted_probs = None
        predicted_labels = None

        self._quantum_instance = self._quantum_instance if quantum_instance is None else quantum_instance
        for i, batch in enumerate(batches):
            if len(batches) > 0:
                logger.debug('Predicting batch {}'.format(i))
            batch_probs, batch_labels = self._get_prediction(batch, params)
            if predicted_probs is None and predicted_labels is None:
                predicted_probs = batch_probs
                predicted_labels = batch_labels
            else:
                np.concatenate((predicted_probs, batch_probs))
                np.concatenate((predicted_labels, batch_labels))
        self._ret['predicted_probs'] = predicted_probs
        self._ret['predicted_labels'] = predicted_labels
        return predicted_probs, predicted_labels

    def _run(self):
        self.train(self._training_dataset[0], self._training_dataset[1])

        if self._test_dataset is not None:
            self.test(self._test_dataset[0], self._test_dataset[1])

        if self._datapoints is not None:
            predicted_probs, predicted_labels = self.predict(self._datapoints)
            self._ret['predicted_classes'] = map_label_to_class_name(predicted_labels,
                                                                     self._label_to_class)
        return self._ret

    def get_optimal_cost(self):
        if 'opt_params' not in self._ret:
            raise AquaError("Cannot return optimal cost before running the algorithm to find optimal params.")
        return self._ret['min_val']

    def get_optimal_circuit(self):
        if 'opt_params' not in self._ret:
            raise AquaError("Cannot find optimal circuit before running the algorithm to find optimal params.")
        return self._var_form.construct_circuit(self._ret['opt_params'])

    def get_optimal_vector(self):
        from qiskit.aqua.utils.run_circuits import find_regs_by_name

        if 'opt_params' not in self._ret:
            raise AquaError("Cannot find optimal vector before running the algorithm to find optimal params.")
        qc = self.get_optimal_circuit()
        if self._quantum_instance.is_statevector:
            ret = self._quantum_instance.execute(qc)
            self._ret['min_vector'] = ret.get_statevector(qc, decimals=16)
        else:
            c = ClassicalRegister(qc.width(), name='c')
            q = find_regs_by_name(qc, 'q')
            qc.add_register(c)
            qc.barrier(q)
            qc.measure(q, c)
            ret = self._quantum_instance.execute(qc)
            self._ret['min_vector'] = ret.get_counts(qc)
        return self._ret['min_vector']

    @property
    def optimal_params(self):
        if 'opt_params' not in self._ret:
            raise AquaError("Cannot find optimal params before running the algorithm.")
        return self._ret['opt_params']

    @property
    def ret(self):
        return self._ret

    @ret.setter
    def ret(self, new_value):
        self._ret = new_value

    @property
    def label_to_class(self):
        return self._label_to_class

    @property
    def class_to_label(self):
        return self._class_to_label

    def load_model(self, file_path):
        model_npz = np.load(file_path, allow_pickle=True)
        self._ret['opt_params'] = model_npz['opt_params']

    def save_model(self, file_path):
        model = {'opt_params': self._ret['opt_params']}
        np.savez(file_path, **model)

    @property
    def test_dataset(self):
        return self._test_dataset

    @property
    def training_dataset(self):
        return self._training_dataset

    @property
    def datapoints(self):
        return self._datapoints
