import copy
import warnings

import numpy as np
import scipy as sp
import sklearn
import sklearn.preprocessing
from lime import explanation
from lime.lime_tabular import LimeTabularExplainer, TableDomainMapper
from pyDOE2 import lhs
from scipy.stats.distributions import norm


class LimeWrapper(LimeTabularExplainer):
    def data_inverse(self, data_row, num_samples, sampling_method, random_state=None):
        """Generates a neighborhood around a prediction.
        For numerical features, perturb them by sampling from a Normal(0,1) and
        doing the inverse operation of mean-centering and scaling, according to
        the means and stds in the training data. For categorical features,
        perturb by sampling according to the training distribution, and making
        a binary feature that is 1 when the value is the same as the instance
        being explained.
        Args:
            data_row: 1d numpy array, corresponding to a row
            num_samples: size of the neighborhood to learn the linear model
            sampling_method: 'gaussian' or 'lhs'
        Returns:
            A tuple (data, inverse), where:
                data: dense num_samples * K matrix, where categorical features
                are encoded with either 0 (not equal to the corresponding value
                in data_row) or 1. The first row is the original instance.
                inverse: same as data, except the categorical features are not
                binary, but categorical (as the original data)
        """
        random_state = random_state or self.random_state
        is_sparse = sp.sparse.issparse(data_row)
        if is_sparse:
            num_cols = data_row.shape[1]
            data = sp.sparse.csr_matrix((num_samples, num_cols), dtype=data_row.dtype)
        else:
            num_cols = data_row.shape[0]
            data = np.zeros((num_samples, num_cols))
        categorical_features = range(num_cols)
        if self.discretizer is None:
            instance_sample = data_row
            scale = self.scaler.scale_
            mean = self.scaler.mean_
            if is_sparse:
                # Perturb only the non-zero values
                non_zero_indexes = data_row.nonzero()[1]
                num_cols = len(non_zero_indexes)
                instance_sample = data_row[:, non_zero_indexes]
                scale = scale[non_zero_indexes]
                mean = mean[non_zero_indexes]

            if sampling_method == "gaussian":
                data = random_state.normal(0, 1, num_samples * num_cols).reshape(num_samples, num_cols)
                data = np.array(data)
            elif sampling_method == "lhs":
                data = lhs(num_cols, samples=num_samples).reshape(num_samples, num_cols)
                means = np.zeros(num_cols)
                stdvs = np.array([1] * num_cols)
                for i in range(num_cols):
                    data[:, i] = norm(loc=means[i], scale=stdvs[i]).ppf(data[:, i])
                data = np.array(data)
            else:
                warnings.warn(
                    """Invalid input for sampling_method.
                                 Defaulting to Gaussian sampling.""",
                    UserWarning,
                )
                data = random_state.normal(0, 1, num_samples * num_cols).reshape(num_samples, num_cols)
                data = np.array(data)

            if self.sample_around_instance:
                data = data * scale + instance_sample
            else:
                data = data * scale + mean
            if is_sparse:
                if num_cols == 0:
                    data = sp.sparse.csr_matrix((num_samples, data_row.shape[1]), dtype=data_row.dtype)
                else:
                    indexes = np.tile(non_zero_indexes, num_samples)
                    indptr = np.array(range(0, len(non_zero_indexes) * (num_samples + 1), len(non_zero_indexes)))
                    data_1d_shape = data.shape[0] * data.shape[1]
                    data_1d = data.reshape(data_1d_shape)
                    data = sp.sparse.csr_matrix((data_1d, indexes, indptr), shape=(num_samples, data_row.shape[1]))
            categorical_features = self.categorical_features
            first_row = data_row
        else:
            first_row = self.discretizer.discretize(data_row)
        data[0] = data_row.copy()
        inverse = data.copy()
        for column in categorical_features:
            values = self.feature_values[column]
            freqs = self.feature_frequencies[column]
            inverse_column = random_state.choice(values, size=num_samples, replace=True, p=freqs)
            binary_column = (inverse_column == first_row[column]).astype(int)
            binary_column[0] = 1
            inverse_column[0] = data[0, column]
            data[:, column] = binary_column
            inverse[:, column] = inverse_column
        if self.discretizer is not None:
            inverse[1:] = self.discretizer.undiscretize(inverse[1:])
        inverse[0] = data_row
        return data, inverse

    def explain_instance(
        self,
        data_row,
        predict_fn,
        labels=(1,),
        top_labels=None,
        num_features=1000,
        num_samples=5000,
        distance_metric="euclidean",
        model_regressor=None,
        sampling_method="gaussian",
    ):
        if sp.sparse.issparse(data_row) and not sp.sparse.isspmatrix_csr(data_row):
            # Preventative code: if sparse, convert to csr format if not in csr format already
            data_row = data_row.tocsr()
        data, inverse = self.data_inverse(data_row, num_samples, sampling_method)

        self.data = data
        _, nf = self.data.shape
        self.inverse = inverse

        if sp.sparse.issparse(data):
            # Note in sparse case we don't subtract mean since data would become dense
            scaled_data = data.multiply(self.scaler.scale_)
            # Multiplying with csr matrix can return a coo sparse matrix
            if not sp.sparse.isspmatrix_csr(scaled_data):
                scaled_data = scaled_data.tocsr()
        else:
            scaled_data = data

        distances = sklearn.metrics.pairwise_distances(
            scaled_data, scaled_data[0].reshape(1, -1), metric=distance_metric
        ).ravel()

        yss = predict_fn(inverse)
        self.yss = yss
        self.scaled_data = scaled_data

        # for classification, the model needs to provide a list of tuples - classes
        # along with prediction probabilities
        if self.mode == "classification":
            if len(yss.shape) == 1:
                raise NotImplementedError(
                    "LIME does not currently support "
                    "classifier models without probability "
                    "scores. If this conflicts with your "
                    "use case, please let us know: "
                    "https://github.com/datascienceinc/lime/issues/16"
                )
            elif len(yss.shape) == 2:
                if self.class_names is None:
                    self.class_names = [str(x) for x in range(yss[0].shape[0])]
                else:
                    self.class_names = list(self.class_names)
                if not np.allclose(yss.sum(axis=1), 1.0):
                    warnings.warn(
                        """
                    Prediction probabilties do not sum to 1, and
                    thus does not constitute a probability space.
                    Check that you classifier outputs probabilities
                    (Not log probabilities, or actual class predictions).
                    """
                    )
            else:
                raise ValueError("Your model outputs " "arrays with {} dimensions".format(len(yss.shape)))

        # for regression, the output should be a one-dimensional array of predictions
        else:
            try:
                if len(yss.shape) != 1 and len(yss[0].shape) == 1:
                    yss = np.array([v[0] for v in yss])
                assert isinstance(yss, np.ndarray) and len(yss.shape) == 1
            except AssertionError:
                raise ValueError(
                    "Your model needs to output single-dimensional \
                    numpyarrays, not arrays of {} dimensions".format(
                        yss.shape
                    )
                )

            predicted_value = yss[0]
            min_y = min(yss)
            max_y = max(yss)

            # add a dimension to be compatible with downstream machinery
            yss = yss[:, np.newaxis]

        feature_names = copy.deepcopy(self.feature_names)
        if feature_names is None:
            feature_names = [str(x) for x in range(data_row.shape[0])]

        if sp.sparse.issparse(data_row):
            values = self.convert_and_round(data_row.data)
            feature_indexes = data_row.indices
        else:
            values = self.convert_and_round(data_row)
            feature_indexes = None

        for i in self.categorical_features:
            if self.discretizer is not None and i in self.discretizer.lambdas:
                continue
            name = int(data_row[i])
            if i in self.categorical_names:
                name = self.categorical_names[i][name]
            feature_names[i] = "%s=%s" % (feature_names[i], name)
            values[i] = "True"
        categorical_features = self.categorical_features

        discretized_feature_names = None
        if self.discretizer is not None:
            categorical_features = range(data.shape[1])
            discretized_instance = self.discretizer.discretize(data_row)
            discretized_feature_names = copy.deepcopy(feature_names)
            for f in self.discretizer.names:
                discretized_feature_names[f] = self.discretizer.names[f][int(discretized_instance[f])]

        domain_mapper = TableDomainMapper(
            feature_names,
            values,
            scaled_data[0],
            categorical_features=categorical_features,
            discretized_feature_names=discretized_feature_names,
            feature_indexes=feature_indexes,
        )
        ret_exp = explanation.Explanation(domain_mapper, mode=self.mode, class_names=self.class_names)
        ret_exp.score = {}
        ret_exp.local_pred = {}
        if self.mode == "classification":
            ret_exp.predict_proba = yss[0]
            if top_labels:
                labels = np.argsort(yss[0])[-top_labels:]
                ret_exp.top_labels = list(labels)
                ret_exp.top_labels.reverse()
        else:
            ret_exp.predicted_value = predicted_value
            ret_exp.min_value = min_y
            ret_exp.max_value = max_y
            labels = [0]

        lines = []
        for label in labels:
            (
                ret_exp.intercept[label],
                ret_exp.local_exp[label],
                ret_exp.score[label],
                ret_exp.local_pred[label],
            ) = self.base.explain_instance_with_data(
                scaled_data,
                yss,
                distances,
                label,
                num_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection,
            )
            b = ret_exp.intercept[label]
            exp = ret_exp.local_exp[label]
            exp = sorted(exp, key=lambda x: x[0])
            w = np.zeros(nf)
            for e in exp:
                w[e[0]] = e[1]
            lines.append((w, b - 0.5))

        self.data = data
        self.data_pred = np.argmax(yss, axis=-1)
        self.exp = ret_exp
        return lines[0]
