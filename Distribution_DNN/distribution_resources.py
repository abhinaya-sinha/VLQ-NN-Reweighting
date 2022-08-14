import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import math

class distribution_resources:
    def E(dist, dist_type):
        # returns expectation value of the distribution
        if dist_type == 'gaussian':
            return dist[0]
        elif dist_type == 'poisson':
            return dist
        elif dist_type == 'weibull':
            k = dist[0]
            l = dist[1]
            E = l*math.gamma(1+1.0/k)
            return E
        elif dist_type == 'continuous bernoulli':
            if dist == 0.5:
                return 0.5
            else:
                E = dist/(2*dist-1)+1/(2*math.atanh(1-2*dist))
                return E
        else:
            raise Exception('that probability distribution is not supported.')
    
    def sigma(dist, dist_type):
        # returns standard deviation of the distribution
        if dist_type == 'gaussian':
            return dist[1]
        elif dist_type == 'poisson':
            return math.sqrt(dist)
        elif dist_type == 'weibull':
            k = dist[0]
            l = dist[1]
            sigma = math.abs(l*math.sqrt(math.gamma(1+2/k)-(math.gamma(1+1/k))**2))
            return sigma
        elif dist_type == 'continuous bernoulli':
            if dist == 0.5:
                return math.sqrt(1/12)
            else:
                sigma = math.sqrt((1-dist)*dist/(1-2*dist)**2 + 1/(2*math.atanh(1-2*dist))**2)
                return sigma
        else:
            raise Exception('that probability distribution is not supported.')

    def distribution_plotter_gaussian(ax, dist):
        mu = distribution_resources.E(dist, 'gaussian')
        sigma = distribution_resources.sigma(dist, 'gaussian')
        x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
        ax.plot(x, stats.norm.pdf(x, mu, sigma))
        return ax

    def distribution_plotter_poisson(ax, dist):
        mu = distribution_resources.E(dist, 'poisson')
        sigma = distribution_resources.sigma(dist, 'poisson')
        x = np.linspace(mu - 3*sigma, mu + 3*sigma, 6*sigma)
        ax.plot(x, stats.poisson.pmf(x, mu), 'bo', ms=8)
        ax.vlines(x, 0, stats.poisson.pmf(x, mu), colors='b', lw=5, alpha=0.5)
        return ax

    def distribution_plotter_weibull(ax, dist):
        c = dist[0]/dist[1]
        x = np.linspace(stats.weibull_min.ppf(0.01, c),stats.weibull_min.ppf(0.99, c), 100)
        ax.plot(x, stats.weibull_min.pdf(x, c))
        return ax

    def distribution_plotter_cont_bernoulli(ax, dist):
        mu = distribution_resources.E(dist, 'continuous bernoulli')
        sigma = distribution_resources.sigma(dist, 'continuous bernoulli')
        x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
        if dist == 0.5:
            C = 2
        else:
            C = 2*math.atanh(1-2*dist)/(1-2*dist)
        ax.plot(x, [C*np.exp(dist, xi)*np.exp((1-dist),(1-xi)) for xi in x])
        return ax