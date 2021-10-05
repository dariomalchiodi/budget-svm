import streamlit as st

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

from optimization import GurobiSolver
from kernel import LinearKernel, GaussianKernel

@st.cache
def generate_data():
    n = 50
    a = 1.5
    b = 0.3

    X = np.random.random(n)
    y = a * X + b + np.random.random(n) / 2

    return X, y

st.markdown('''# Budget-based SVR

The basic SVR algorithm solves the following optimization problem:''')

st.latex(r'''\min \frac{1}{2} ||w||^2 + C \sum(\xi_i + \hat\xi_i)''')

st.markdown('under the constraints')

st.latex(r'''y_i - \epsilon - \xi_i \leq w \cdot x_i + b
          \leq y_i + \epsilon + \hat\xi_i''')
st.latex(r'\xi_i, \hat\xi_i \geq 0')

st.markdown('The Wolfe dual is')

st.latex(r'''\min_{\alpha, \hat\alpha} \frac{1}{2} \sum_{i, j}
  (\alpha_i - \hat\alpha_i) (\alpha_j - \hat\alpha_j) k(x_i, x_j)
  - \sum_i (\alpha_i - \hat\alpha_i) y_i
  + \sum_i (\alpha_i + \hat\alpha_i) \epsilon ''')

st.markdown('under the constraints')

st.latex(r'\sum_i (\alpha_i - \hat\alpha_i) = 0')
st.latex(r'0 \leq \alpha_i, \hat\alpha_i \leq C')

X, y = generate_data()

s = GurobiSolver()

C = st.sidebar.slider('C', min_value=.1, max_value=10., step=.01, value=1.)

k = st.sidebar.selectbox('Kernel', (LinearKernel(), GaussianKernel(3),
               GaussianKernel(.1), GaussianKernel(.03)))

epsilon = st.sidebar.slider(r'$\epsilon$', min_value=.01, max_value=1.,
                    step=.01, value=.1)
#k = GaussianKernel(sigma=st.sidebar.slider('sigma', min_value=0.01,
#                                           max_value=100, step=.01, value=1.))
alpha, alpha_hat = s.solve(X, y, C, k, epsilon)

with st.expander(label='Optimization results'):
    col_alpha, col_alpha_hat = st.columns(2)
    with col_alpha:
        st.latex(r'\alpha:')
        alpha_df = pd.DataFrame(alpha)
        alpha_df.index.name = 'alpha'
        st.dataframe(alpha_df)
    with col_alpha_hat:
        st.latex(r'\hat\alpha:')
        alpha_hat

sv = [(a - a_h, x) for a, a_h, x in zip(alpha, alpha_hat, X) if a - a_h != 0]

def feature_dot(x_new, sv, kernel):
    return sum([a * kernel.compute(x, x_new) for (a, x) in sv])

#w = sum([feature_dot(x, sv, k) for x in X])

b_values = [y_ - feature_dot(x, sv, k) - epsilon
            for a, x, y_ in zip(alpha, X, y) if 0 < a < C]
b_values += [y_ - feature_dot(x, sv, k) + epsilon
             for a_h, x, y_ in zip(alpha_hat, X, y) if 0 < a_h < C]

b = np.mean(b_values)

regression = lambda x: feature_dot(x, sv, k) + b

x_values = np.linspace(min(X), max(X), 500)

fig, ax = plt.subplots()
ax.scatter(X, y)
ax.plot(x_values, [regression(x) for x in x_values])
ax.plot(x_values, [regression(x) + epsilon for x in x_values],
        'k--', linewidth=.5)
ax.plot(x_values, [regression(x) - epsilon for x in x_values],
        'k--', linewidth=.5)

st.markdown('The budget-based version, instead, focuses on')

st.latex(r'''\min_{\alpha, \hat\alpha, \gamma} \frac{1}{2} \sum_{i, j}
  (\alpha_i - \hat\alpha_i) (\alpha_j - \hat\alpha_j) k(x_i, x_j)
  - \sum_i (\alpha_i - \hat\alpha_i) y_i
  + \sum_i (\alpha_i + \hat\alpha_i) \epsilon
  + \gamma B''')

st.markdown('under the constraints')

st.latex(r'\sum_i (\alpha_i - \hat\alpha_i) = 0')
st.latex(r'\alpha_i - \gamma \leq C')
st.latex(r'\hat\alpha_i - \gamma \leq C')
st.latex(r'\alpha_i, \hat\alpha_i, \gamma \geq 0')

st.markdown('The KKT conditions are')

st.latex(r'\alpha_i(w \cdot x_i + b - y_i + \epsilon + \xi_i) = 0')
st.latex(r'\hat\alpha_i(y_i + \epsilon + \hat\xi_i - w \cdot x_i - b) = 0')
st.latex(r'\beta_i \xi_i = 0, \hat\beta_i \hat\xi_i = 0')
st.latex(r'\gamma B \sum(\xi_i + \hat\xi_i) = 0')

st.markdown(r'''So that if the optimal value of $\gamma$ is zero, $b$ is found
   as usual, otherwise $b = y_i - \epsilon - w \cdot x_i$ with $i$ such that
   $\alpha_i < C + \gamma$, still considering optimal values of variables.
   Note that in both cases $b$ can be found considering $i$ such that
   $\alpha_i < C + \gamma$.
   Similar considerations hold for the hatted set of variables.''')

B = st.sidebar.slider('B', min_value=.01, max_value=3., step=.1, value=1.)

alpha, alpha_hat, gamma = s.solve(X, y, C, k, epsilon, budget=B)

with st.expander(label='Optimization results'):
    col_alpha, col_alpha_hat, col_gamma = st.columns(3)
    with col_alpha:
        st.latex(r'\alpha:')
        alpha_df = pd.DataFrame(alpha)
        alpha_df.index.name = 'alpha'
        st.dataframe(alpha_df)
    with col_alpha_hat:
        st.latex(r'\hat\alpha:')
        alpha_hat
    with col_gamma:
        st.latex(rf'\gamma: {gamma}')

budget_sv = [(a - a_h, x) for a, a_h, x
             in zip(alpha, alpha_hat, X) if a - a_h != 0]


budget_b_values = [y_ - feature_dot(x, sv, k) - epsilon
            for a, x, y_ in zip(alpha, X, y) if 0 < a < C + gamma]
budget_b_values += [y_ - feature_dot(x, sv, k) + epsilon
             for a_h, x, y_ in zip(alpha_hat, X, y) if 0 < a_h < C + gamma]

budget_b = np.mean(b_values)

budget_regression = lambda x: feature_dot(x, budget_sv, k) + budget_b

ax.plot(x_values, [budget_regression(x) + epsilon for x in x_values],
        'r:', linewidth=.5)
ax.plot(x_values, [budget_regression(x) for x in x_values], 'r')
ax.plot(x_values, [budget_regression(x) - epsilon for x in x_values],
        'r:', linewidth=.5)
st.pyplot(fig)
