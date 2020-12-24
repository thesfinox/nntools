'''
Plot utilities module - plot several types of graphics and save the figures

Author(s)
-----------

Riccardo Finotello (riccardo.finotello@gmail.com)
'''

import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set()


def ratio(x, y, base=(640,480), dpi=72):
    '''
    Define the ratio of the plots.
    
    Required arguments:
        x: column ratio,
        y: row ratio.
        
    Optional arguments:
        base: 1x1 default ratio in pixels,
        dpi:  dots per inch.
        
    Returns:
        the size of the figure.
    '''
    
    return (base[0] * x / dpi, base[1] * y / dpi)
    
def savefig(filename, fig, root='.', show=False, save_pdf=True, save_png=False, dpi=72):
    '''
    Save a Matplotlib figure to file.
    
    Required arguments:
        filename: the name of the saved file in the root directory (do not add an extension),
        fig:      the Matplotlib figure object.
        
    Optional arguments:
        root:     root directory,
        show:     show the plot inline (bool),
        save_pdf: save in PDF format,
        save_png: save in PNG format.
    '''
    
    # save the figure to file (PDF and PNG)
    plt.tight_layout()
    if save_pdf:
        os.makedirs(root, exist_ok=True)
        fig.savefig(os.path.join(root, filename + '.pdf'), dpi=dpi, format='pdf')
    if save_png:
        os.makedirs(root, exist_ok=True)
        fig.savefig(os.path.join(root, filename + '.png'), dpi=dpi, format='png')
    
    # show if interactive
    if show:
        plt.show()
    
    # release memory
    fig.clf()
    plt.close(fig)
    
    
def plot_univariate(data,
                    x=None,
                    hue=None,
                    weights=None,
                    stat='count',
                    bins='auto',
                    discrete=False,
                    title=None,
                    xlabel=None,
                    ylabel=None,
                    logx=False,
                    logy=False,
                    alpha=0.5,
                    fill=True,
                    colour='C0',
                    palette='deep',
                    palette_colours=None,
                    bbox_to_anchor=(1.0, 0.0),
                    anchor='lower left',
                    base_size=(640, 480),
                    base_ratio=(1,1),
                    subplots=None,
                    return_ax=False,
                    out_name=None,
                    dpi=72,
                    **kwargs
                   ):
    '''
    Plot a univariate distribution as histogram.
    
    Required arguments:
        data: the data to plot (can be a dict if multiple series on the same plot).
    
    Optional arguments:
        x:               name of the series to plot in case data is a container,
        hue:             name of the series to distinguish colours in data,
        weights:         weights of the bins if data is a weighted distribution,
        stat:            'count', 'frequency', 'density', 'probability' (https://seaborn.pydata.org/generated/seaborn.hist plot.html#seaborn.histplot),
        bins:            number of of bins or criterium to create them (https://numpy.org/doc/stable/reference/generated/nu mpy.histogram_bin_edges.html#numpy.histogram_bin_edges),
        discrete:        if True, then binwidth=1 and values are centred on the ticks,
        title:           title of the plot,
        xlabel:          x-axis label,
        ylabel:          y-axis label,
        logx:            use log scale on x-axis,
        logy:            use log scale on y-axis,
        alpha:           transparency factor,
        fill:            fill the histogram or leave the white interior,
        colour:          the colour of the distribution (if data is a series),
        palette:         colour palette if data is a container (use 'train_test' if data is in the train:validation:test format),
        palette_colours: dictionary mapping for the colours,
        bbox_to_anchor:  position of the legend,
        anchor:          anchor of the legend,
        base_size:       tuple with the size of the output,
        base_ratio:      ratio of the output plot (tuple),
        subplots:        pass a tuple of (figure, axis) to use existing axis,
        return_ax:       return the axis object if requested,
        out_name:        name of the ouput (without extension),
        **kwargs:        additional arguments to pass to savefig.
        
    Returns:
        the axis (if requested).
        
    E.g.:
    
        data = {'training': ...,
                'validation': ...,
                'test': ...
               }
               
        The keys of the dictionary will then be used as legend.
    '''
    
    # plot the distribution
    if subplots is not None:
        fig, ax = subplots
    else:
        X, Y    = base_ratio
        fig, ax = plt.subplots(1, 1, figsize=ratio(base_ratio[0], base_ratio[1], base=base_size, dpi=dpi), dpi=dpi)
        
    # create a palette
    legend          = False
    
    if isinstance(data, dict):
        data = pd.DataFrame(data)
            
    if isinstance(data, pd.DataFrame):
        # override colour and legend
        if x is not None:
            legend = True
        colour = None
        
        # create palette
        if palette == 'train_test' or 'training' in data.columns:
            palette_list    = ['tab:blue', 'tab:red', 'tab:green']
            palette_colours = {key: palette_list[n] for n, key in enumerate(data.columns)}
        else:
            if palette_colours is None:
                palette_colours = sns.color_palette(palette, n_colors=len(data.columns))
            
    # plot the data
    histplot = {'data':      data,
                'x':         x,
                'hue':       hue,
                'weights':   weights,
                'stat':      stat,
                'bins':      bins,
                'discrete':  discrete,
                'fill':      fill,
                'alpha':     alpha,
                'palette':   palette_colours,
                'color':     colour,
                'legend':    legend,
                'log_scale': (logx, logy)
               }
    sns.histplot(**histplot, ax=ax)
    
    ax.set_title(title)
    ax.set_xlabel(xlabel if xlabel is not None else x)
    ax.set_ylabel(ylabel=ylabel if ylabel is not None else stat)
    
    # set legend
    if isinstance(data, pd.DataFrame) and x is None:
        ax.legend(labels=data.columns, bbox_to_anchor=bbox_to_anchor, loc=anchor)

    # save the figure
    if out_name is not None:
        savefig(out_name, fig, dpi=dpi, **kwargs)
        
    if return_ax:
        return ax
    
    
def plot_bivariate(data,
                   x,
                   y,
                   hue=None,
                   size=None,
                   style=None,
                   title=None,
                   xlabel=None,
                   ylabel=None,
                   logx=False,
                   logy=False,
                   alpha=None,
                   colour='C0',
                   palette='deep',
                   palette_colours=None,
                   markers=True,
                   markers_dict=None,
                   legend='auto',
                   bbox_to_anchor=(1.0, 0.0),
                   anchor='lower left',
                   base_size=(640, 480),
                   base_ratio=(1,1),
                   subplots=None,
                   return_ax=False,
                   out_name=None,
                   dpi=72,
                   **kwargs
                  ):
    '''
    Plot a bivariate distribution as scatterplot.
    
    Required arguments:
        data: the data to plot (can be a dict if multiple series on the same plot),
        x:    name of the series to plot in the x-axis,
        y:    name of the series to plot in the y-axis (it can be a list).
    
    Optional arguments:
        hue:             name of the series to distinguish colours in data,
        size:            name of the series to distinguish the size of the markers,
        style:           name of the series to distinguish the style of the markers,
        title:           title of the plot,
        xlabel:          x-axis label,
        ylabel:          y-axis label,
        logx:            use log scale on x-axis,
        logy:            use log scale on y-axis,
        alpha:           transparency factor,
        colour:          color for single series plots,
        palette:         colour palette if data is a container,
        palette_colours: dictionary mapping for the colours,
        markers:         whether to use markers to differentiate the points,
        markers_dict:    dictionary map between the variables and the markers,
        legend:          type of legend to draw (https://seaborn.pydata.org/generated/seaborn.scatterplot.html#seaborn.scatterplot),
        bbox_to_anchor:  position of the legend,
        anchor:          anchor of the legend,
        base_size:       tuple with the size of the output,
        base_ratio:      ratio of the output plot (tuple),
        subplots:        pass a tuple of (figure, axis) to use existing axis,
        return_ax:       return the axis object if requested,
        out_name:        name of the ouput (without extension),
        **kwargs:        additional arguments to pass to savefig.
        
    Returns:
        the axis (if requested).
        
    E.g.:
    
        data = {'x': ...,
                'training': ...,
                'validation': ...,
                'test': ...
               }
               
        The keys of the dictionary will then be used as legend.
    '''
    
    # plot the distribution
    if subplots is not None:
        fig, ax = subplots
    else:
        X, Y    = base_ratio
        fig, ax = plt.subplots(1, 1, figsize=ratio(base_ratio[0], base_ratio[1], base=base_size, dpi=dpi), dpi=dpi)
        
    if isinstance(data, dict):
        data = pd.DataFrame(data)
        
    # case 1: list of variables
    if isinstance(y, list):
        
        if 'training' in y or 'train' in y:
            palette_colours = {'training':    'tab:blue',
                               'train':       'tab:blue',
                               'validation':  'tab:red',
                               'val':         'tab:red',
                               'development': 'tab:red',
                               'dev':         'tab:red',
                               'test':        'tab:green'
                              }
            
            if markers:
                markers_dict = {'training':    '.',
                                'train':       '.',
                                'validation':  'x',
                                'val':         'x',
                                'development': 'x',
                                'dev':         'x',
                                'test':        '+'
                               }
        
        else:
            if palette_colours is None:
                palette_colours = sns.color_palette(palette, n_colors=len(y))
            if not isinstance(palette_colours, dict):
                palette_colours = {key: palette_colours[n] for n, key in enumerate(y)}
                
        # plot the data
        for var in y:
            scatterplot = {'data':      data,
                           'x':         data.index if x is None else x,
                           'y':         var,
                           'hue':       hue,
                           'size':      size,
                           'style':     style,
                           'alpha':     alpha,
                           'markers':   markers,
                           'marker':    markers_dict[var] if markers_dict is not None else None,
                           'palette':   palette,
                           'color':     palette_colours[var] if palette_colours is not None else None,
                           'legend':    legend
                          }
            sns.scatterplot(**scatterplot, ax=ax)
        
    # case 2: y is not a list
    else:
        scatterplot = {'data':      data,
                       'x':         data.index if x is None else x,
                       'y':         y,
                       'hue':       hue,
                       'size':      size,
                       'style':     style,
                       'alpha':     alpha,
                       'markers':   markers,
                       'color':     colour,
                       'legend':    legend
                      }
        sns.scatterplot(**scatterplot, ax=ax)
            
    # set properties of the plot
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')
    
    # set legend
    if isinstance(y, list) and len(y) > 1:
        ax.legend(labels=y, bbox_to_anchor=bbox_to_anchor, loc=anchor)

    # save the figure
    if out_name is not None:
        savefig(out_name, fig, dpi=dpi, **kwargs)
        
    if return_ax:
        return ax
    
    
def plot_line(data,
              y,
              x=None,
              hue=None,
              size=None,
              style=None,
              title=None,
              xlabel=None,
              ylabel=None,
              logx=False,
              logy=False,
              alpha=None,
              colour='C0',
              palette='deep',
              palette_colours=None,
              dashes=True,
              dashes_dict=None,
              legend='auto',
              bbox_to_anchor=(1.0, 0.0),
              anchor='lower left',
              base_size=(640, 480),
              base_ratio=(1,1),
              subplots=None,
              return_ax=False,
              out_name=None,
              dpi=72,
              **kwargs
             ):
    '''
    Plot a function or lineplot.
    
    Required arguments:
        data:           dict or dataframe with the data,
        y:              name of the series to plot in the y-axis (it can be a list).
    
    Optional arguments:
        x:               name of the series to plot in the x-axis,
        hue:             name of the series to distinguish colours in data,
        size:            name of the series to distinguish the size of the markers,
        style:           name of the series to distinguish the style of the markers,
        title:           title of the plot,
        xlabel:          x-axis label,
        ylabel:          y-axis label,
        logx:            use log scale on x-axis,
        logy:            use log scale on y-axis,
        alpha:           transparency factor,
        colour:          color for single series plots,
        palette:         colour palette if data is a container (use 'train_test' if data is in the train:validation:test format),
        palette_colours: dictionary mapping for the colours,
        dashes:          whether to use different dashes for the lines,
        dashes_dict:     dictionary map between variables and dashes,
        legend:          type of legend to draw (https://seaborn.pydata.org/generated/seaborn.scatterplot.html#seaborn.scatterplot),
        bbox_to_anchor:  position of the legend,
        anchor:          anchor of the legend,
        base_size:       tuple with the size of the output,
        base_ratio:      ratio of the output plot (tuple),
        subplots:        pass a tuple of (figure, axis) to use existing axis,
        return_ax:       return the axis object if requested,
        out_name:        name of the ouput (without extension),
        **kwargs:        additional arguments to pass to savefig.
        
    Returns:
        the axis (if requested).
        
    E.g.:
    
        data = {'x': ...,
                'training': ...,
                'validation': ...,
                'test': ...
               }
               
        The keys of the dictionary will then be used as legend.
    '''
    
    # plot the distribution
    if subplots is not None:
        fig, ax = subplots
    else:
        X, Y    = base_ratio
        fig, ax = plt.subplots(1, 1, figsize=ratio(base_ratio[0], base_ratio[1], base=base_size, dpi=dpi), dpi=dpi)
        
    if isinstance(data, dict):
        data = pd.DataFrame(data)
        
    # case 1: list of variables
    if isinstance(y, list):
        
        if 'training' in y or 'train' in y:
            palette_colours = {'training':    'tab:blue',
                               'train':       'tab:blue',
                               'validation':  'tab:red',
                               'val':         'tab:red',
                               'development': 'tab:red',
                               'dev':         'tab:red',
                               'test':        'tab:green'
                              }
            
            if dashes:
                dashes_dict = {'training':    '-',
                               'train':       '-',
                               'validation':  '--',
                               'val':         '--',
                               'development': '--',
                               'dev':         '--',
                               'test':        '-.'
                              }
        
        else:
            if palette_colours is None:
                palette_colours = sns.color_palette(palette, n_colors=len(y))
            if not isinstance(palette_colours, dict):
                palette_colours = {key: palette_colours[n] for n, key in enumerate(y)}
                
        # plot the data
        for var in y:
            lineplot = {'data':      data,
                        'x':         x,
                        'y':         var,
                        'hue':       hue,
                        'size':      size,
                        'style':     style,
                        'alpha':     alpha,
                        'dashes':    dashes,
                        'linestyle': dashes_dict[var] if dashes_dict is not None else None,
                        'palette':   palette,
                        'color':     palette_colours[var] if palette_colours is not None else None,
                        'legend':    legend
                       }
            sns.lineplot(**lineplot, ax=ax)
        
    # case 2: y is not a list
    else:
        lineplot = {'data':      data,
                    'x':         x,
                    'y':         y,
                    'hue':       hue,
                    'size':      size,
                    'style':     style,
                    'alpha':     alpha,
                    'dashes':    False,
                    'color':     colour,
                    'legend':    legend
                   }
        sns.lineplot(**lineplot, ax=ax)
            
    # set properties of the plot
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')
    
    # set legend
    if isinstance(y, list) and len(y) > 1:
        ax.legend(labels=y, bbox_to_anchor=bbox_to_anchor, loc=anchor)

    # save the figure
    if out_name is not None:
        savefig(out_name, fig, dpi=dpi, **kwargs)
        
    if return_ax:
        return ax
    
    
def plot_catbox(data,
                x=None,
                y=None,
                hue=None,
                order=None,
                orient=None,
                title=None,
                xlabel=None,
                ylabel=None,
                palette='deep',
                bbox_to_anchor=(1.0, 0.0),
                anchor='lower left',
                base_size=(640, 480),
                base_ratio=(1,1),
                subplots=None,
                return_ax=False,
                out_name=None,
                dpi=72,
                **kwargs
               ):
    '''
    Plot a categorical boxplot.
    
    Required arguments:
        data: the data to plot (can be a dict if multiple series on the same plot).
    
    Optional arguments:
        x:              name of the categorical data,
        y:              values of the categories,
        hue:            name of the series to distinguish colours in data,
        order:          ordered list in which to plot the categorical data (otherwise it is inferred by the data),
        orient:         'v' or 'h' for vertical or horizontal orientations,
        title:          title of the plot,
        xlabel:         x-axis label,
        ylabel:         y-axis label,
        alpha:          transparency factor,
        palette:        colour palette,
        bbox_to_anchor: position of the legend,
        anchor:         anchor of the legend,
        base_size:      tuple with the size of the output,
        base_ratio:     ratio of the output plot (tuple),
        subplots:       pass a tuple of (figure, axis) to use existing axis,
        return_ax:      return the axis object if requested,
        out_name:       name of the ouput (without extension),
        **kwargs:       additional arguments to pass to savefig.
        
    Returns:
        the axis (if requested).
    '''
    
    # plot the distribution
    if subplots is not None:
        fig, ax = subplots
    else:
        X, Y    = base_ratio
        fig, ax = plt.subplots(1, 1, figsize=ratio(base_ratio[0], base_ratio[1], base=base_size, dpi=dpi), dpi=dpi)
        
    # create a palette
    palette_colours = None
    
    if isinstance(data, dict):
        data = pd.DataFrame(data)
            
    # plot the data
    boxplot = {'data':      data,
               'x':         x,
               'y':         y,
               'order':     order,
               'orient':    orient,
               'hue':       hue,
               'palette':   palette,
              }
    sns.boxplot(**boxplot, ax=ax)
    
    ax.set_title(title)
    ax.set_xlabel(xlabel if xlabel is not None else x)
    ax.set_ylabel(ylabel if ylabel is not None else y)

    # save the figure
    if out_name is not None:
        savefig(out_name, fig, dpi=dpi, **kwargs)
        
    if return_ax:
        return ax
    

def plot_corr(data,
              title=None,
              cmap='RdBu_r',
              bbox_to_anchor=(1.0, 0.0),
              anchor='lower left',
              base_size=(640, 480),
              base_ratio=(1,1),
              subplots=None,
              return_ax=False,
              out_name=None,
              dpi=72,
              **kwargs
             ):
    '''
    Plot the correlation matrix.
    
    Required arguments:
        data: the data to plot.
    
    Optional arguments:
        out_name:       name of the ouput (without extension),
        root:           root of the save location,
        title:          title of the plot,
        cmap:           the color of the heatmap,
        bbox_to_anchor: position of the legend,
        anchor:         anchor of the legend,
        base_size:      tuple with the size of the output,
        base_ratio:     ratio of the output plot (tuple),
        subplots:       pass a tuple of (figure, axis) to use existing axis,
        return_ax:      return the axis object if requested,
        out_name:       name of the ouput (without extension),
        **kwargs:       additional arguments to pass to savefig.
        
    Returns:
        the axis (if requested).
    '''
    
    # plot the distribution
    if subplots is not None:
        fig, ax = subplots
    else:
        X, Y    = base_ratio
        fig, ax = plt.subplots(1, 1, figsize=ratio(base_ratio[0], base_ratio[1], base=base_size, dpi=dpi), dpi=dpi)

    # compute correlations and plot
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)
        
    corr_mat = data.corr()
        
    sns.heatmap(corr_mat,
                center=0.0,
                cmap=cmap,
                ax=ax
               )
    
    ax.set_title(title)
    ax.set_xticks(np.arange(len(corr_mat.columns)) + 0.5)
    ax.set_yticks(np.arange(len(corr_mat.columns)) + 0.5)
    ax.set_xticklabels(corr_mat.columns, rotation=90, va='top', ha='center')
    ax.set_yticklabels(corr_mat.columns, va='center', ha='right')

    # save the figure
    if out_name is not None:
        savefig(out_name, fig, dpi=dpi, **kwargs)
        
    if return_ax:
        return ax
        
        
def plot_history(history,
                 metric,
                 orient='index',
                 validation=False,
                 title='Metric Function',
                 xlabel='epochs',
                 ylabel='metric',
                 logx=False,
                 logy=False,
                 smooth=None,
                 smooth_window=10,
                 alpha=None,
                 legend='auto',
                 bbox_to_anchor=(1.0, 0.0),
                 anchor='lower left',
                 base_size=(640, 480),
                 base_ratio=(1,1),
                 subplots=None,
                 return_ax=False,
                 out_name=None,
                 dpi=72,
                 **kwargs
                ):
    '''
    Plot the a metric or loss function from a Keras history.
    
    Required arguments:
        history: location of the JSON file containing the history of the training,
        metric:  the name of the metric to plot or list of metrics,.
    
    Optional arguments:
        orient:         orientation of the JSON file (for Pandas),
        out_name:       name of the ouput (without extension),
        root:           root of the save location,
        validation:     plot validation loss,
        title:          title of the plot,
        logx:           use log scale on x-axis,
        logy:           use log scale on y-axis,
        smooth:         smoothen the behaviour by considering a moving average,
        smooth_window:  width of the smoothing window,
        alpha:          transparency factor,
        legend:         type of legend to draw (https://seaborn.pydata.org/generated/seaborn.scatterplot.html#seaborn.scatterplot),
        bbox_to_anchor: position of the legend,
        anchor:         anchor of the legend,
        base_size:      tuple with the size of the output,
        base_ratio:     ratio of the output plot (tuple),
        subplots:       pass a tuple of (figure, axis) to use existing axis,
        return_ax:      return the axis object if requested,
        out_name:       name of the ouput (without extension),
        **kwargs:       additional arguments to pass to savefig.
    '''
    
    # open JSON file
    if isinstance(history, str):
        hst = pd.read_json(history, orient=orient)
    else:
        hst = pd.DataFrame(history)
        
    # select data
    if not isinstance(metric, list):
        metric = [metric]

    # plot the function
    if subplots is not None:
        fig, ax = subplots
    else:
        X, Y    = base_ratio
        fig, ax = plt.subplots(1, 1, figsize=ratio(base_ratio[0], base_ratio[1], base=base_size, dpi=dpi), dpi=dpi)
        
    # define the palettes
    plot_palette_train = ['tab:blue', 'tab:cyan',   'tab:purple', 'tab:grey']
    plot_palette_val   = ['tab:red',  'tab:orange', 'tab:pink',   'tab:olive']

    # plot the metrics
    plot_metrics = []
    plot_palette = {}
    plot_dashes  = {}
    plot_hst     = {}
    for n, m in enumerate(metric):
        plot_metrics.append(m)
        plot_palette[m] = plot_palette_train[n]
        plot_dashes[m]  = '-'
        
        # smooth if requested
        if smooth:
            if 1 < smooth_window < np.shape(hst[m])[0]:
                plot_hst[m] = np.convolve(hst[m], np.ones(smooth_window) / smooth_window, mode='valid')
        else:
            plot_hst[m] = hst[m]
        
        if validation:
            plot_metrics.append('val_' + m)
            plot_palette['val_' + m] = plot_palette_val[n]
            plot_dashes['val_' + m] = '--'
        
            # smooth if requested
            if smooth:
                if 1 < smooth_window < np.shape(hst['val_' + m])[0]:
                    plot_hst['val_' + m] = np.convolve(hst['val_' + m], np.ones(smooth_window) / smooth_window, mode='valid')
            else:
                plot_hst['val_' + m] = hst['val_' + m]
    
    # plot the new metrics
    plot_hst = pd.DataFrame(plot_hst)
    for var in plot_metrics:
        lineplot = {'data':      plot_hst,
                    'x':         plot_hst.index,
                    'y':         var,
                    'alpha':     alpha,
                    'linestyle': plot_dashes[var],
                    'color':     plot_palette[var],
                    'legend':    legend
                   }
        sns.lineplot(**lineplot, ax=ax)
        
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if len(plot_metrics) > 1:
        ax.legend(labels=plot_metrics, bbox_to_anchor=bbox_to_anchor, loc=anchor)
    
    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')

    # save the figure
    if out_name is not None:
        savefig(out_name, fig, dpi=dpi, **kwargs)
        
    if return_ax:
        return ax