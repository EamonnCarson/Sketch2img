# Final Project: Sketch2Img
##### CS 182: Neural Networks, Spring 2019
##### Authors: David Wang, Danny Kim, Eamonn Carson, Kyle Kovach

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
$$
\newcommand{\given}[1][]{\:#1\vert\:}
\newcommand{\Prob}[1]{\mathbb{P}\left( #1 \right)}
\newcommand{\Var}{\mathrm{Var}}
\newcommand{\Cov}{\mathrm{Cov}}
\newcommand{\Expect}[1]{\mathrm{E}\left[ #1 \right]}
\newcommand{\divides}[1]{\,\left\vert\, #1 \right.}
\newcommand{\Note}[1]{\textnormal{ #1 }}
\newcommand{\Naturals}{\mathbb{N}}
\newcommand{\Posints}{\mathbb{Z}_{\geq 0}}
\newcommand{\Reals}{\mathbb{R}}
\newcommand{\Rationals}{\mathbb{Q}}
\newcommand{\float}{\mathrm{float}}
\newcommand{\E}[1]{\times10^{#1}}
\newcommand{\abs}[1]{\left\lvert #1 \right\rvert}
\newcommand{\norm}[1]{\left\lVert #1 \right\rVert}
\newcommand{\bigO}[1]{\mathcal{O}\left( #1 \right)}
\newcommand{\suchthat}{\textnormal{ such that }}
\newcommand{\vspan}[1]{\textnormal{span}\left\{#1\right\}}
\newcommand{\domatrix}[1]{
    \begin{pmatrix} 
        #1 
    \end{pmatrix}
}
$$

Content goes here


$$ hello $$

$$ 
\begin{align*}
    L(D) &= 
        L_{\textnormal{GAN}}(D, G) 
        + L_{\textnormal{AC}}(D)
    \\
    L(G) &= 
        L_{\textnormal{GAN}}(G) 
        - L_{\textnormal{AC}}(G)
        + L_{\textnormal{sup}}(G)
        + L_{\textnormal{p}}(G)
        + L_{\textnormal{div}}(G)
        \\
        &
        \\
    L_{\textnormal{GAN}}(D, G) &= 
    \mathrm{E}_{Y \sim P_{\textnormal{image}}}\left[ \log D(y) \right]
    + \mathrm{E}_{Y \sim P_{\textnormal{sketch}},\, z \sim P_z}\left[ \log (1 - D(G(x,z)) \right]
    \\
    L_{\textnormal{AC}} &= \Expect{\log P\left(C = c \given y \right)}
    \\
    L_{\textnormal{sup}} &= \norm{G(x, z) - y}_1
    \\
    L_{\textnormal{p}} &= \sum_{i} \lambda_p \norm{\phi_i\left( G(x,z) \right) - \phi_i\left( y \right)}_1
    \\
    L_{\textnormal{div}} &= -\lambda_{\textnormal{div}} \norm{G(x, z_1) - G(x, z_2)}_1
    \\
\end{align*}
$$

\begin{align*}
    L(D) &= 
        L_{\textnormal{GAN}}(D, G) 
        + L_{\textnormal{AC}}(D)
    \\
    L(G) &= 
        L_{\textnormal{GAN}}(G) 
        - L_{\textnormal{AC}}(G)
        + L_{\textnormal{sup}}(G)
        + L_{\textnormal{p}}(G)
        + L_{\textnormal{div}}(G)
        \\
        &
        \\
    L_{\textnormal{GAN}}(D, G) &= 
    \mathrm{E}_{Y \sim P_{\textnormal{image}}}\left[ \log D(y) \right]
    + \mathrm{E}_{Y \sim P_{\textnormal{sketch}},\, z \sim P_z}\left[ \log (1 - D(G(x,z)) \right]
    \\
    L_{\textnormal{AC}} &= \Expect{\log P\left(C = c \given y \right)}
    \\
    L_{\textnormal{sup}} &= \norm{G(x, z) - y}_1
    \\
    L_{\textnormal{p}} &= \sum_{i} \lambda_p \norm{\phi_i\left( G(x,z) \right) - \phi_i\left( y \right)}_1
    \\
    L_{\textnormal{div}} &= -\lambda_{\textnormal{div}} \norm{G(x, z_1) - G(x, z_2)}_1
    \\
\end{align*}

\[\begin{align*}
    L(D) &= 
        L_{\textnormal{GAN}}(D, G) 
        + L_{\textnormal{AC}}(D)
    \\
    L(G) &= 
        L_{\textnormal{GAN}}(G) 
        - L_{\textnormal{AC}}(G)
        + L_{\textnormal{sup}}(G)
        + L_{\textnormal{p}}(G)
        + L_{\textnormal{div}}(G)
        \\
        &
        \\
    L_{\textnormal{GAN}}(D, G) &= 
    \mathrm{E}_{Y \sim P_{\textnormal{image}}}\left[ \log D(y) \right]
    + \mathrm{E}_{Y \sim P_{\textnormal{sketch}},\, z \sim P_z}\left[ \log (1 - D(G(x,z)) \right]
    \\
    L_{\textnormal{AC}} &= \Expect{\log P\left(C = c \given y \right)}
    \\
    L_{\textnormal{sup}} &= \norm{G(x, z) - y}_1
    \\
    L_{\textnormal{p}} &= \sum_{i} \lambda_p \norm{\phi_i\left( G(x,z) \right) - \phi_i\left( y \right)}_1
    \\
    L_{\textnormal{div}} &= -\lambda_{\textnormal{div}} \norm{G(x, z_1) - G(x, z_2)}_1
    \\
\end{align*}\]

