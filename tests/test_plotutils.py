import unittest
from nntools import *

# create placeholder data
data_1 = {'training':   np.random.normal(size=(100000,)),
          'validation': np.random.normal(size=(100000,)),
          'test':       np.random.normal(size=(100000,)),
         }

data_2 = {'placeholder_1': np.random.normal(size=(100000,)),
          'placeholder_2': np.random.normal(size=(100000,))
         }

data_3 = {'values': np.random.normal(size=(100,)),
          'type':   np.random.choice(['A', 'B', 'C'], size=(100,)),
          'diff':   np.random.choice(['type_1', 'type_2'], size=(100,))
         }

data_4 = {'x':          np.linspace(0, 6, 100),
          'training':   np.sin(np.linspace(0, 6, 100)),
          'validation': 0.9 * np.sin(np.linspace(0, 6, 100)),
          'test':       1.1 * np.sin(np.linspace(0, 6, 100)),
          'style':      (np.linspace(0, 6, 100) < 2),
          'size':       (np.linspace(0, 6, 100) > 2)
         }

data_5 = np.random.normal(size=(25, 10))

data_6 = {'loss':     np.exp(-np.linspace(0, 6, 100)),
          'val_loss': [n + np.random.uniform() for n in np.exp(-np.linspace(0, 6, 100))],
          'mse':      [0.9 * n for n in np.exp(-np.linspace(0, 6, 100))],
          'val_mse':  [0.9 * n + np.random.uniform() for n in np.exp(-np.linspace(0, 6, 100))],
          'mae':      [0.8 * n for n in np.exp(-np.linspace(0, 6, 100))],
          'val_mae':  [0.8 * n + np.random.uniform() for n in np.exp(-np.linspace(0, 6, 100))],
          'lr':       1e-3 * np.exp(-np.linspace(0, 6, 100))
         }

# test class
class TestPlots(unittest.TestCase):
    
    def test_univariate(self):
        
        print('Testing univariate distribution plots...')
        
        plot_univariate(data_1,
                        title='Training - Validation - Test',
                        logy=True,
                        out_name='hist_train_val_test_dict',
                        root='./img',
                        save_pdf=True
                       )
        
        plot_univariate(data_1,
                        x='validation',
                        title='Training - Validation - Test',
                        logy=True,
                        out_name='hist_train_val_test_single',
                        root='./img',
                        save_pdf=True
                       )
        
        plot_univariate(data_2,
                        title='Test Data',
                        xlabel='data',
                        out_name='hist_non_train_val_test_dict',
                        root='./img',
                        save_pdf=True
                       )
        
        plot_univariate(data_2['placeholder_1'],
                        title='Test Data',
                        xlabel='data',
                        out_name='hist_non_train_val_test_single',
                        root='./img',
                        save_pdf=True
                       )
        
        plot_univariate(data_3,
                        x='values',
                        hue='type',
                        title='Hue Histogram',
                        xlabel='values',
                        out_name='hist_categorical',
                        root='./img',
                        save_pdf=True
                       )
        
    def test_bivariate(self):
        
        print('Testing bivariate distribution plots...')
        
        plot_bivariate(data_4,
                       x='x',
                       y='training',
                       title='Sine Plot',
                       xlabel='data',
                       ylabel='sin(x)',
                       out_name='scatter',
                       root='./img',
                       save_pdf=True
                      )
        
        plot_bivariate(data_4,
                       x='x',
                       y=['training', 'validation', 'test'],
                       title='Sine Plot',
                       xlabel='data',
                       ylabel='sin(x)',
                       out_name='scatter_list',
                       root='./img',
                       save_pdf=True
                      )
        
        plot_bivariate(data_4,
                       x='x',
                       y=['training', 'validation'],
                       title='Sine Plot',
                       xlabel='data',
                       ylabel='sin(x)',
                       out_name='scatter_list_notest',
                       root='./img',
                       save_pdf=True
                      )
        
        plot_bivariate(data_4,
                       x='x',
                       y='training',
                       style='style',
                       size='size',
                       title='Sine Plot',
                       xlabel='data',
                       ylabel='sin(x)',
                       out_name='scatter_list_style',
                       root='./img',
                       save_pdf=True
                      )
        
        plot_line(data_4,
                  x='x',
                  y='training',
                  title='Sine Plot',
                  xlabel='data',
                  ylabel='sin(x)',
                  out_name='lineplot',
                  root='./img',
                  save_pdf=True
                 )
        
        plot_line(data_4,
                  x='x',
                  y=['training', 'validation'],
                  title='Sine Plot',
                  xlabel='data',
                  ylabel='sin(x)',
                  dashes=False,
                  out_name='lineplot_nodashes',
                  root='./img',
                  save_pdf=True
                 )
        
        plot_line(data_4,
                  x='x',
                  y=['training', 'validation', 'test'],
                  title='Sine Plot',
                  xlabel='data',
                  ylabel='sin(x)',
                  out_name='lineplot_list',
                  root='./img',
                  save_pdf=True
                 )
        
        plot_line(data_4,
                  y=['training', 'validation'],
                  title='Sine Plot',
                  xlabel='data',
                  ylabel='sin(x)',
                  out_name='lineplot_list_nox',
                  root='./img',
                  save_pdf=True
                 )
        
        plot_line(data_4,
                  x='x',
                  y='training',
                  style='style',
                  size='size',
                  title='Sine Plot',
                  xlabel='data',
                  ylabel='sin(x)',
                  out_name='lineplot_list_style',
                  root='./img',
                  save_pdf=True
                 )
        
    def test_catbox(self):
        
        print('Testing categorical boxplots...')
        
        plot_catbox(data_3,
                    x='type',
                    y='values',
                    title='Categorical Boxplot',
                    xlabel='type',
                    ylabel='values',
                    out_name='cat_boxplot',
                    root='./img',
                    save_pdf=True
                   )
        
        plot_catbox(data_3,
                    x='type',
                    y='values',
                    order=['A', 'B', 'C'],
                    title='Categorical Boxplot',
                    xlabel='type',
                    ylabel='values',
                    out_name='cat_boxplot_ordered',
                    root='./img',
                    save_pdf=True
                   )
        
    def test_heatmap(self):
        
        print('Testing heatmap for correlation matrices...')
        
        plot_corr(data_5,
                  title='Correlation Matrix',
                  out_name='corr_mat',
                  root='./img',
                  save_pdf=True
                 )
        
    def test_print_metrics(self):
        
        print('Testing printing methods for metric functions...')
        
        plot_history(data_6,
                     'loss',
                     validation=True,
                     title='Loss Function',
                     xlabel='epochs',
                     ylabel='loss',
                     logx=False,
                     logy=False,
                     smooth=None,
                     alpha=None,
                     out_name='loss_function',
                     save_pdf=True,
                     root='./img'
                    )
        
        plot_history(data_6,
                     ['loss', 'mse'],
                     validation=True,
                     title='Loss and Metric Function',
                     xlabel='epochs',
                     ylabel='loss / metric',
                     logx=False,
                     logy=False,
                     smooth=True,
                     smooth_window=10,
                     alpha=None,
                     out_name='loss_and_metric_function',
                     save_pdf=True,
                     root='./img'
                    )
        
        plot_history(data_6,
                     ['loss', 'mse', 'mae'],
                     validation=True,
                     title='Loss and Metric Functions',
                     xlabel='epochs',
                     ylabel='loss / metric',
                     logx=False,
                     logy=True,
                     smooth=True,
                     smooth_window=10,
                     alpha=None,
                     out_name='loss_and_metric_functions',
                     save_pdf=True,
                     root='./img'
                    )
        
        plot_history(data_6,
                     'lr',
                     validation=False,
                     title='Learning Rate',
                     xlabel='epochs',
                     ylabel=None,
                     logx=False,
                     logy=True,
                     smooth=False,
                     smooth_window=10,
                     alpha=1.0,
                     out_name='learning_rate',
                     save_pdf=True,
                     root='./img'
                    )
        
        
if __name__ == '__main__':
    unittest.main()