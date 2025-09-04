import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, classification_report,
    confusion_matrix, precision_score, recall_score, f1_score
)
from sklearn.calibration import calibration_curve
import os

from vehicle_collision_prediction.config import ModelEvalConfig


class ModelEvaluator:
    def __init__(self, config: ModelEvalConfig):
        self.config: ModelEvalConfig = config
        self.app_color_palette = [
            'rgba(99, 110, 250, 0.8)',   # Blue
            'rgba(239, 85, 59, 0.8)',    # Red/Orange-Red
            'rgba(0, 204, 150, 0.8)',    # Green
            'rgba(171, 99, 250, 0.8)',   # Purple
            'rgba(255, 161, 90, 0.8)',   # Orange
            'rgba(25, 211, 243, 0.8)',   # Cyan
            'rgba(255, 102, 146, 0.8)',  # Pink
            'rgba(182, 232, 128, 0.8)',  # Light Green
            'rgba(255, 151, 255, 0.8)',  # Magenta
            'rgba(254, 203, 82, 0.8)'    # Yellow
        ]

    def evaluate_model(self, model, X_test, y_test, output_dir):
        """
        Comprehensive model evaluation with plots and metrics
        
        Parameters:
        model: Trained model with predict and predict_proba methods
        X_test: Test features
        y_test: Test targets
        output_dir: Directory to save plots
        
        Returns:
        dict: Evaluation metrics
        """
        # Create output directory for plots
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Get predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_proba)
        
        # Generate plots
        self._create_confusion_matrix_plot(y_test, y_pred, os.path.join(plots_dir, "confusion_matrix.html"))
        self._create_precision_recall_curves(y_test, y_proba, os.path.join(plots_dir, "precision_recall_curves.html"))
        self._create_class_distribution_plot(y_test, y_pred, os.path.join(plots_dir, "class_distribution.html"))
        self._create_probability_distribution_plot(y_proba, y_test, os.path.join(plots_dir, "probability_distributions.html"))
        self._create_calibration_plot(y_test, y_proba, os.path.join(plots_dir, "calibration_plot.html"))
        
        # Feature importance plot if available
        if hasattr(model, 'get_feature_importance'):
            feature_importance = model.get_feature_importance()
            if feature_importance:
                self._create_feature_importance_plot(feature_importance, os.path.join(plots_dir, "feature_importance.html"))
        
        return metrics

    def _calculate_metrics(self, y_true, y_pred, y_proba):
        """Calculate comprehensive evaluation metrics"""
        # Get unique classes
        classes = np.unique(y_true)
        
        # Primary metric: Macro PR-AUC
        pr_aucs = []
        for i, cls in enumerate(classes):
            y_binary = (y_true == cls).astype(int)
            if i < y_proba.shape[1]:
                y_prob_cls = y_proba[:, i]
                pr_auc = average_precision_score(y_binary, y_prob_cls)
                pr_aucs.append(pr_auc)
        
        macro_pr_auc = np.mean(pr_aucs)
        
        # Secondary metrics
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        precision_macro = precision_score(y_true, y_pred, average='macro')
        recall_macro = recall_score(y_true, y_pred, average='macro')
        
        # Per-class metrics
        class_report = classification_report(y_true, y_pred, output_dict=True)
        
        return {
            'macro_pr_auc': macro_pr_auc,
            'individual_pr_aucs': {f'class_{cls}': pr_aucs[i] for i, cls in enumerate(classes) if i < len(pr_aucs)},
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'classification_report': class_report
        }

    def _create_confusion_matrix_plot(self, y_true, y_pred, filepath):
        """Create confusion matrix heatmap"""
        cm = confusion_matrix(y_true, y_pred)
        
        fig = px.imshow(
            cm,
            title="Confusion Matrix",
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=[f"Class {i}" for i in range(cm.shape[1])],
            y=[f"Class {i}" for i in range(cm.shape[0])],
            text_auto=True
        )
        
        self._apply_theme(fig)
        fig.write_html(filepath, include_plotlyjs=True, config={'responsive': True, 'displayModeBar': False})

    def _create_precision_recall_curves(self, y_true, y_proba, filepath):
        """Create precision-recall curves for each class"""
        classes = np.unique(y_true)
        
        fig = go.Figure()
        
        for i, cls in enumerate(classes):
            if i < y_proba.shape[1]:
                y_binary = (y_true == cls).astype(int)
                precision, recall, _ = precision_recall_curve(y_binary, y_proba[:, i])
                pr_auc = average_precision_score(y_binary, y_proba[:, i])
                
                fig.add_trace(go.Scatter(
                    x=recall, y=precision,
                    mode='lines',
                    name=f'Class {cls} (PR-AUC: {pr_auc:.3f})',
                    line=dict(color=self.app_color_palette[i % len(self.app_color_palette)])
                ))
        
        fig.update_layout(
            title="Precision-Recall Curves by Class",
            xaxis_title="Recall",
            yaxis_title="Precision"
        )
        
        self._apply_theme(fig)
        fig.write_html(filepath, include_plotlyjs=True, config={'responsive': True, 'displayModeBar': False})

    def _create_class_distribution_plot(self, y_true, y_pred, filepath):
        """Create class distribution comparison plot"""
        classes = np.unique(np.concatenate([y_true, y_pred]))
        
        true_counts = pd.Series(y_true).value_counts().sort_index()
        pred_counts = pd.Series(y_pred).value_counts().reindex(true_counts.index, fill_value=0)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=[f"Class {cls}" for cls in true_counts.index],
            y=true_counts.values,
            name="Actual",
            marker_color=self.app_color_palette[0]
        ))
        
        fig.add_trace(go.Bar(
            x=[f"Class {cls}" for cls in pred_counts.index],
            y=pred_counts.values,
            name="Predicted",
            marker_color=self.app_color_palette[1]
        ))
        
        fig.update_layout(
            title="Class Distribution: Actual vs Predicted",
            xaxis_title="Class",
            yaxis_title="Count",
            barmode='group'
        )
        
        self._apply_theme(fig)
        fig.write_html(filepath, include_plotlyjs=True, config={'responsive': True, 'displayModeBar': False})

    def _create_probability_distribution_plot(self, y_proba, y_true, filepath):
        """Create probability distribution plots by class"""
        classes = np.unique(y_true)
        
        fig = make_subplots(
            rows=1, cols=len(classes),
            subplot_titles=[f"Class {cls} Probability Distribution" for cls in classes]
        )
        
        for i, cls in enumerate(classes):
            if i < y_proba.shape[1]:
                # Split probabilities by true class
                for j, true_cls in enumerate(classes):
                    mask = y_true == true_cls
                    if np.any(mask):
                        fig.add_trace(
                            go.Histogram(
                                x=y_proba[mask, i],
                                name=f"True Class {true_cls}",
                                opacity=0.7,
                                legendgroup=f"true_{true_cls}",
                                showlegend=(i == 0),
                                marker_color=self.app_color_palette[j % len(self.app_color_palette)]
                            ),
                            row=1, col=i+1
                        )
        
        fig.update_layout(title="Probability Distributions by True Class")
        self._apply_theme(fig)
        fig.write_html(filepath, include_plotlyjs=True, config={'responsive': True, 'displayModeBar': False})

    def _create_calibration_plot(self, y_true, y_proba, filepath):
        """Create calibration plot for probability calibration assessment"""
        classes = np.unique(y_true)
        
        fig = go.Figure()
        
        # Add perfect calibration line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Perfect Calibration',
            line=dict(dash='dash', color='gray')
        ))
        
        for i, cls in enumerate(classes):
            if i < y_proba.shape[1]:
                y_binary = (y_true == cls).astype(int)
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    y_binary, y_proba[:, i], n_bins=10
                )
                
                fig.add_trace(go.Scatter(
                    x=mean_predicted_value,
                    y=fraction_of_positives,
                    mode='lines+markers',
                    name=f'Class {cls}',
                    line=dict(color=self.app_color_palette[i % len(self.app_color_palette)])
                ))
        
        fig.update_layout(
            title="Calibration Plot",
            xaxis_title="Mean Predicted Probability",
            yaxis_title="Fraction of Positives"
        )
        
        self._apply_theme(fig)
        fig.write_html(filepath, include_plotlyjs=True, config={'responsive': True, 'displayModeBar': False})

    def _create_feature_importance_plot(self, feature_importance, filepath):
        """Create feature importance plot"""
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        features, importances = zip(*sorted_features[:20])  # Top 20 features
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=list(importances),
            y=list(features),
            orientation='h',
            marker_color=self.app_color_palette[0]
        ))
        
        fig.update_layout(
            title="Top 20 Feature Importances",
            xaxis_title="Importance",
            yaxis_title="Features"
        )
        
        self._apply_theme(fig)
        fig.write_html(filepath, include_plotlyjs=True, config={'responsive': True, 'displayModeBar': False})

    def _apply_theme(self, fig):
        """Apply consistent theme to plots"""
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#8B5CF6', size=12),
            title_font=dict(color='#7C3AED', size=16),
            xaxis=dict(
                gridcolor='rgba(139,92,246,0.2)',
                zerolinecolor='rgba(139,92,246,0.3)',
                tickfont=dict(color='#8B5CF6', size=11),
                title_font=dict(color='#7C3AED', size=12)
            ),
            yaxis=dict(
                gridcolor='rgba(139,92,246,0.2)',
                zerolinecolor='rgba(139,92,246,0.3)',
                tickfont=dict(color='#8B5CF6', size=11),
                title_font=dict(color='#7C3AED', size=12)
            ),
            legend=dict(font=dict(color='#8B5CF6', size=11))
        )
