clear all
close all

addpath Data
addpath Algorithms
addpath Tools

warning off

%% Load  the small MNIST dataset 

% subset of MNIST with only 0 and 1
load('Data/dataset_subMNIST.mat')

%% Useful dimensions

Nx = size(X_train, 1) ;
Ny = size(X_train, 2) ;
N = Nx*Ny ;
L = size(X_train, 3) ;

%% Parameters for training

% training set as an N x L matrix (each column contains an image)
X_train_mat = reshape(X_train, N, L) ;

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TO COMPLETE
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Lipschitz constant of smooth term
% Frobenius norm used to compute squared root of the sum of the square of
% every entity in the matrix
beta = (norm(X_train_mat, "fro")^2)/L;
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% regularization parameters
lambda = 1e-3 ;
delta = 1e-1 ;

% initialization
winit = zeros(N,1) ;

%% Function to check classifier

% binary classifier (returns 1 if answer to question is "yes", and -1 for "no")
d=@(x,w) sign( x'*w ); 
% size of the test dataset
L_test = length(Y_test) ; 
% binary classifier applied to test set
d_test =@(w) d(reshape(X_test, N, L_test), w);

%% Forward-Backward algorithm

Stop_norm = 1e-4 ; 
Stop_crit = 1e-4 ;
ItMax = 10000 ;

[w, perc_error, crit, time] = ...
    FB(winit, X_train_mat, Y_train, lambda, delta, beta, ...
                    d_test, Y_test, ItMax, Stop_norm, Stop_crit) ;

save("resultFB_small_MNIST");


%% Load  the full MNIST dataset 

% full MNIST dataset
load('Data/dataset_MNIST.mat')

%% Useful dimensions

Nx = size(X_train, 1) ;
Ny = size(X_train, 2) ;
N = Nx*Ny ;
L = size(X_train, 3) ;

%% Parameters for training

% training set as an N x L matrix (each column contains an image)
X_train_mat = reshape(X_train, N, L) ;

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TO COMPLETE
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Lipschitz constant of smooth term
% Frobenius norm used to compute squared root of the sum of the square of
% every entity in the matrix
beta = (norm(X_train_mat, "fro")^2)/L;
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% regularization parameters
lambda = 1e-3 ;
delta = 1e-1 ;

% initialization
winit = zeros(N,1) ;

%% Function to check classifier

% binary classifier (returns 1 if answer to question is "yes", and -1 for "no")
d=@(x,w) sign( x'*w ); 
% size of the test dataset
L_test = length(Y_test) ; 
% binary classifier applied to test set
d_test =@(w) d(reshape(X_test, N, L_test), w);

%% Forward-Backward algorithm

Stop_norm = 1e-4 ; 
Stop_crit = 1e-4 ;
ItMax = 10000 ;

[w, perc_error, crit, time] = ...
    FB(winit, X_train_mat, Y_train, lambda, delta, beta, ...
                    d_test, Y_test, ItMax, Stop_norm, Stop_crit) ;

save("resultFB_full_MNIST");