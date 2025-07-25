import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)

# Import models
from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.linear_model import LogisticRegression

class ModelEvaluator:
    """
    A comprehensive class for evaluating machine learning models.

    This class handles data preprocessing, hyperparameter tuning using GridSearchCV,
    model training, performance evaluation, and visualization of results,
    including feature importance and prediction plots.
    """
    def __init__(self):
        """
        Initializes the ModelEvaluator with a predefined dictionary of models and
        their hyperparameter search spaces for both regression and classification tasks.
        """
        self.model_params = {
            'xgb': {
                'regression': {
                    'model': XGBRegressor(random_state=42),
                    'params': {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [3, 5, 7],
                        'learning_rate': [0.01, 0.05, 0.1],
                        'subsample': [0.8, 0.9, 1.0],
                        'colsample_bytree': [0.8, 0.9, 1.0]
                    }
                },
                'classification': {
                    'model': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
                    'params': {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [3, 5, 7],
                        'learning_rate': [0.05, 0.1, 0.2],
                        'scale_pos_weight': [1, 5, 10]
                    }
                }
            },
            'gbdt': {
                'regression': {
                    'model': GradientBoostingRegressor(random_state=42),
                    'params': {
                        'n_estimators': [100, 200, 300],
                        'learning_rate': [0.01, 0.05, 0.1],
                        'max_depth': [3, 5, 7],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4]
                    }
                },
                'classification': {
                    'model': GradientBoostingClassifier(random_state=42),
                    'params': {
                        'n_estimators': [100, 200, 300],
                        'learning_rate': [0.05, 0.1, 0.2],
                        'max_depth': [3, 5, 7]
                    }
                }
            },
            'knn': {
                'regression': {
                    'model': KNeighborsRegressor(),
                    'params': {
                        'n_neighbors': [3, 5, 7, 9],
                        'weights': ['uniform', 'distance'],
                        'p': [1, 2]
                    }
                },
                'classification': {
                    'model': KNeighborsClassifier(),
                    'params': {
                        'n_neighbors': [3, 5, 7, 9],
                        'weights': ['uniform', 'distance']
                    }
                }
            },
            'rf': {
                'regression': {
                    'model': RandomForestRegressor(random_state=42),
                    'params': {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [10, 20, None],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4],
                        'max_features': ['auto', 'sqrt']
                    }
                },
                'classification': {
                    'model': RandomForestClassifier(random_state=42),
                    'params': {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [10, 15, 20, None],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4],
                        'class_weight': ['balanced', None]
                    }
                }
            },
            'svm': {
                'regression': {
                    'model': SVR(),
                    'params': {
                        'kernel': ['rbf', 'linear'],
                        'C': [0.1, 1, 10, 100],
                        'gamma': ['scale', 'auto']
                    }
                },
                'classification': {
                    'model': SVC(probability=True, random_state=42),
                    'params': {
                        'kernel': ['rbf', 'linear'],
                        'C': [0.1, 1, 10, 100],
                        'gamma': ['scale', 'auto'],
                        'class_weight': ['balanced', None]
                    }
                }
            },
            'lr': {
                'classification': {
                    'model': LogisticRegression(random_state=42, max_iter=2000, solver='liblinear'),
                    'params': {
                        'C': [10, 50, 100, 150],
                        'penalty': ['l1', 'l2'],
                        'solver': ['liblinear', 'saga'],
                        'class_weight': ['balanced', None]
                    }
                }
            }
        }
        self.reset_results()

    def reset_results(self):
        """
        Resets the instance variables to None.
        
        This method is called to clear the state of the evaluator before a new
        evaluation run, preventing data leakage from previous runs.
        """
        self.best_model = None
        self.best_params = None
        self.metrics = None
        self.y_pred = None
        self.X_train, self.X_test = None, None
        self.y_train, self.y_test = None, None
        self.feature_names = None
        self.grid_search = None

    def evaluate(self, X, y, model_name, task_type='regression', test_size=0.2, random_state=42, result_path=None, save_results=True):
        """
        Evaluates a specified model's performance.

        Args:
            X (pd.DataFrame or np.ndarray): The feature matrix.
            y (pd.Series or np.ndarray): The target vector.
            model_name (str): The name of the model to evaluate (key in self.model_params).
            task_type (str): The type of task, either 'regression' or 'classification'.
            test_size (float): The proportion of the dataset to allocate to the test split.
            random_state (int): The seed used by the random number generator.
            result_path (str): The directory path to save evaluation results.
            save_results (bool): If True, saves metrics, predictions, and feature importance.
        
        Returns:
            ModelEvaluator: The instance itself, allowing for method chaining.
        """
        self.reset_results()

        if result_path is None:
            result_path = './results'
        os.makedirs(result_path, exist_ok=True)

        # Store original feature names if X is a DataFrame
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        
        # Preprocess the data (handling missing values, encoding, and scaling will be added here)
        # For now, we assume the data is clean and ready for modeling.

        # Split data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Get model configuration from the dictionary
        model_config = self.model_params[model_name][task_type]

        # Define scoring metrics based on the task type
        if task_type == 'regression':
            scoring = {'r2': 'r2', 'neg_rmse': 'neg_root_mean_squared_error', 'neg_mae': 'neg_mean_absolute_error'}
            refit_metric = 'r2'
        else:
            scoring = {'accuracy': 'accuracy', 'f1': 'f1_weighted', 'precision': 'precision_weighted', 'recall': 'recall_weighted'}
            refit_metric = 'f1'

        # Set up GridSearchCV to find the best hyperparameters
        self.grid_search = GridSearchCV(
            estimator=model_config['model'],
            param_grid=model_config['params'],
            cv=5,
            scoring=scoring,
            refit=refit_metric,
            n_jobs=-1,  # Use all available CPU cores
            verbose=1
        )
        
        # Fit the model to the training data
        self.grid_search.fit(self.X_train, self.y_train)

        self.best_model = self.grid_search.best_estimator_
        self.best_params = self.grid_search.best_params_

        # Make predictions on both training and test sets
        self.y_train_pred = self.best_model.predict(self.X_train)
        self.y_test_pred = self.best_model.predict(self.X_test)
        
        # Calculate performance metrics
        self.metrics = self._calculate_metrics(task_type)

        # Save results if requested
        if save_results:
            self._save_all_results(model_name, task_type, result_path)
    
        return self

    def _calculate_metrics(self, task_type):
        """Internal helper method to calculate performance metrics."""
        metrics = {}
        best_index = self.grid_search.best_index_
        cv_results = self.grid_search.cv_results_

        if task_type == 'regression':
            metrics.update({
                'cv_r2_mean': cv_results['mean_test_r2'][best_index],
                'cv_r2_std': cv_results['std_test_r2'][best_index],
                'cv_rmse_mean': -cv_results['mean_test_neg_rmse'][best_index],
                'cv_rmse_std': cv_results['std_test_neg_rmse'][best_index],
                'cv_mae_mean': -cv_results['mean_test_neg_mae'][best_index],
                'cv_mae_std': cv_results['std_test_neg_mae'][best_index],
                'test_r2': r2_score(self.y_test, self.y_test_pred),
                'test_rmse': np.sqrt(mean_squared_error(self.y_test, self.y_test_pred)),
                'test_mae': mean_absolute_error(self.y_test, self.y_test_pred)
            })
        else: # Classification
            metrics.update({
                'cv_accuracy': cv_results['mean_test_accuracy'][best_index],
                'cv_f1_weighted': cv_results['mean_test_f1'][best_index],
                'cv_precision_weighted': cv_results['mean_test_precision'][best_index],
                'cv_recall_weighted': cv_results['mean_test_recall'][best_index],
                'test_accuracy': accuracy_score(self.y_test, self.y_test_pred),
                'test_f1_weighted': f1_score(self.y_test, self.y_test_pred, average='weighted'),
                'test_precision_weighted': precision_score(self.y_test, self.y_test_pred, average='weighted'),
                'test_recall_weighted': recall_score(self.y_test, self.y_test_pred, average='weighted')
            })
        return metrics
    
    def _save_all_results(self, model_name, task_type, result_path):
        """Internal helper method to save all evaluation artifacts."""
        # Save predictions to an Excel file
        train_preds_df = pd.DataFrame({'True_Values': self.y_train, 'Predicted_Values': self.y_train_pred})
        test_preds_df = pd.DataFrame({'True_Values': self.y_test, 'Predicted_Values': self.y_test_pred})
        with pd.ExcelWriter(os.path.join(result_path, f"{model_name}_{task_type}_predictions.xlsx")) as writer:
            train_preds_df.to_excel(writer, sheet_name='Train_Predictions', index=False)
            test_preds_df.to_excel(writer, sheet_name='Test_Predictions', index=False)
        
        # Save metrics to a CSV file
        metrics_df = pd.DataFrame([self.metrics])
        metrics_df['model'] = model_name
        metrics_df.to_csv(os.path.join(result_path, f"{model_name}_{task_type}_metrics.csv"), index=False)
        
        # Save best parameters to a CSV file
        params_df = pd.DataFrame([self.best_params])
        params_df['model'] = model_name
        params_df.to_csv(os.path.join(result_path, f"{model_name}_{task_type}_best_params.csv"), index=False)
        
        # Save feature importance if the model supports it
        if hasattr(self.best_model, 'feature_importances_') or isinstance(self.best_model, (XGBClassifier, XGBRegressor)):
            try:
                explainer = shap.TreeExplainer(self.best_model, self.X_train)
                shap_values = explainer.shap_values(self.X_test)
                
                # For multiclass classification, shap_values is a list of arrays. We take the one for the positive class.
                if isinstance(shap_values, list):
                     shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                
                feature_names = self.feature_names if self.feature_names else [f'feature_{i}' for i in range(self.X_test.shape[1])]
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'SHAP_Importance': np.abs(shap_values).mean(axis=0)
                }).sort_values(by='SHAP_Importance', ascending=False)
                
                importance_df.to_csv(os.path.join(result_path, f"{model_name}_{task_type}_feature_importance.csv"), index=False)
            except Exception as e:
                print(f"Could not generate SHAP feature importance for {model_name}. Error: {e}")

    def plot_feature_importance(self, save_path=None):
        """Generates and saves SHAP feature importance plots."""
        if not hasattr(self, 'best_model') or self.best_model is None:
            print("Model not trained yet. Please run evaluate() first.")
            return
        if not save_path:
            return

        try:
            explainer = shap.TreeExplainer(self.best_model, self.X_train)
            shap_values = explainer.shap_values(self.X_test)
            feature_names = self.feature_names if self.feature_names else [f'feature_{i}' for i in range(self.X_test.shape[1])]
            
            # Summary Plot
            shap.summary_plot(shap_values, self.X_test, feature_names=feature_names, show=False)
            plt.title('SHAP Feature Importance Summary', fontsize=14, pad=20)
            plt.savefig(save_path.replace('.tiff', '_summary.tiff'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Bar Plot
            shap.summary_plot(shap_values, self.X_test, feature_names=feature_names, plot_type='bar', show=False)
            plt.title('Mean SHAP Value (Feature Importance)', fontsize=14, pad=20)
            plt.savefig(save_path.replace('.tiff', '_bar.tiff'), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Could not create SHAP plots. Error: {e}")

    def plot_results(self, task_type, save_path=None, title=None):
        """Generates and saves result plots (scatter for regression, heatmap for classification)."""
        if self.y_test is None:
            print("Model not trained yet. Please run evaluate() first.")
            return
        if not save_path:
            return

        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(10, 8))

        if task_type == 'regression':
            ax.scatter(self.y_train, self.y_train_pred, alpha=0.6, edgecolors='k', s=80, label='Training Set')
            ax.scatter(self.y_test, self.y_test_pred, alpha=0.8, edgecolors='k', s=80, label='Test Set')
            min_val = min(self.y_test.min(), self.y_train.min())
            max_val = max(self.y_test.max(), self.y_train.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Fit')
            
            metrics_text = (
                f"CV R²: {self.metrics['cv_r2_mean']:.3f}\n"
                f"CV RMSE: {self.metrics['cv_rmse_mean']:.3f}\n"
                f"Test R²: {self.metrics['test_r2']:.3f}\n"
                f"Test RMSE: {self.metrics['test_rmse']:.3f}"
            )
            ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_xlabel('True Values', fontsize=14)
            ax.set_ylabel('Predicted Values', fontsize=14)
            ax.legend(fontsize=12)
        else: # Classification
            cm = confusion_matrix(self.y_test, self.y_test_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, annot_kws={"size": 16})
            ax.set_xlabel('Predicted Label', fontsize=14)
            ax.set_ylabel('True Label', fontsize=14)

        if title:
            ax.set_title(title, fontsize=16, pad=20)
        
        fig.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

if __name__ == "__main__":
    # --- Configuration ---
    base_path = './mof_evaluation_results'
    excel_path = './MOF-tox-cleaned.xlsx'
    task_type = "regression"  # Can be 'regression' or 'classification'

    # --- Data Loading ---
    if not os.path.exists(excel_path):
        print(f"Error: Data file not found at {excel_path}")
    else:
        data = pd.read_excel(excel_path)
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        
        # --- Model Evaluation Loop ---
        evaluator = ModelEvaluator()
        model_names = list(evaluator.model_params.keys())
        
        for model_name in tqdm(model_names, desc='Training Models'):
            # Skip Logistic Regression for regression tasks
            if task_type == 'regression' and model_name == 'lr':
                print(f"\nSkipping {model_name.upper()} as it is not applicable for regression tasks.")
                continue

            print(f"\n--- Training {model_name.upper()} for {task_type} ---")
            
            # Create a dedicated folder for each model's results
            model_path = os.path.join(base_path, model_name)
            os.makedirs(model_path, exist_ok=True)
            
            try:
                # Evaluate the model
                results = evaluator.evaluate(X, y, model_name=model_name, task_type=task_type,
                                     result_path=model_path)
                
                # Plot feature importance
                evaluator.plot_feature_importance(
                    save_path=os.path.join(model_path, "feature_importance.tiff"))
                
                # Plot fitting results
                evaluator.plot_results(
                    task_type=task_type,
                    save_path=os.path.join(model_path, "fit_results.tiff"),
                    title=f"{model_name.upper()} Model - {task_type.capitalize()} Results"
                )
                
                print(f"\n{model_name.upper()} evaluation finished successfully.")
                print("Best Parameters:", results.best_params)
                print("Performance Metrics:")
                # Pretty print the metrics dictionary
                for key, value in results.metrics.items():
                    print(f"  {key}: {value:.4f}")
    
            except Exception as e:
                print(f"\n--- ERROR: {model_name.upper()} model training failed ---")
                print(f"  Error details: {e}")
                continue
    
        print("\n--- All model training cycles complete! ---")