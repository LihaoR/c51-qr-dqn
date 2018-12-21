#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 23:50:01 2018

@author: lihaoruo
"""

import threading
import numpy as np
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import gym
import random
from atari_wrappers import wrap_deepmind
from time import sleep
from replaymemory import ReplayMemory
GLOBAL_STEP = 0

def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

def process_frame(image):
    image = np.reshape(image,[np.prod(image.shape)]) / 255.0
    return image

#Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

class Q_Network():
    def __init__(self,s_size,a_size,scope,trainer):
        with tf.variable_scope(scope):
            #  quantile regression dqn
            self.quantile = 1.0 / N
            self.cumulative_probabilities = (2.0 * np.arange(N) + 1) / (2.0 * N)
            #  network 
            self.inputs = tf.placeholder(shape=[None,s_size],dtype=tf.float32)
            self.imageIn = tf.reshape(self.inputs,shape=[-1,84,84,1])
            self.conv1 = slim.conv2d(activation_fn=tf.nn.relu,
                                     inputs=self.imageIn,num_outputs=32,
                                     kernel_size=[8,8],stride=[4,4],padding='VALID')
            self.conv2 = slim.conv2d(activation_fn=tf.nn.relu,
                                     inputs=self.conv1,num_outputs=64,
                                     kernel_size=[4,4],stride=[2,2],padding='VALID')
            self.conv3 = slim.conv2d(activation_fn=tf.nn.relu,
                                     inputs=self.conv2,num_outputs=64,
                                     kernel_size=[3,3],stride=[1,1],padding='VALID')
            hidden = slim.fully_connected(slim.flatten(self.conv2),512,activation_fn=tf.nn.relu)
            
            self.out = slim.fully_connected(hidden, a_size * N,
                                            activation_fn=None,
                                            weights_initializer=normalized_columns_initializer(0.1),
                                            biases_initializer=None)
            self.out = tf.reshape(self.out, [-1, a_size, N])
            self.Q   = tf.reduce_sum(self.out * self.quantile, axis=2)
            
            #Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':
                self.actions_q = tf.placeholder(shape=[None, a_size, N], dtype=tf.float32)
                self.q_target  = tf.placeholder(shape=[None, N], dtype=tf.float32)
                self.ISWeights = tf.placeholder(shape=[None, N], dtype=tf.float32)
                
                self.q_actiona = tf.multiply(self.out, self.actions_q)
                self.q_action  = tf.reduce_sum(self.q_actiona, axis=1)
                self.u = tf.abs(self.q_target - self.q_action)
                
                self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.u) * self.ISWeights, axis=1))

                #Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss,local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads,self.grad_norms = tf.clip_by_global_norm(self.gradients,40.0)
                
                #Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads,global_vars))

    def huber(self, u, k):
        abs_u = tf.abs(u)
        return tf.where(abs_u <= k, tf.square(u) * 0.5, k * (abs_u - 0.5 * k))

class Worker():
    def __init__(self,env,name,s_size,a_size,trainer,model_path,global_episodes):
        self.name = "worker_" + str(name)
        self.number = name
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        #Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_Q = Q_Network(s_size, a_size, self.name, trainer)
        self.update_local_ops = update_target_graph('global', self.name)
        self.env = env
        self.replaymemory = ReplayMemory(max_memory)
        
    def train(self,rollout,sess,gamma,ISWeights):
        rollout = np.array(rollout)
        observations      = rollout[:,0]
        actions           = rollout[:,1]
        rewards           = rollout[:,2]
        next_observations = rollout[:,3]
        dones             = rollout[:,4]
        
        Q_target = sess.run(self.local_Q.Q, feed_dict={self.local_Q.inputs:np.vstack(next_observations)})
        actions_ = np.argmax(Q_target, axis=1)
        
        action = np.zeros((batch_size, a_size))
        action_ = np.zeros((batch_size, a_size))
        for i in range(batch_size):
            action[i][actions[i]] = 1
            action_[i][actions_[i]] = 1
        action_now = np.zeros((batch_size, a_size, N))
        action_next = np.zeros((batch_size, a_size, N))
        for i in range(batch_size):
            for j in range(a_size):
                for k in range(N):
                    action_now[i][j][k] = action[i][j]
                    action_next[i][j][k] = action_[i][j]

        q_target = sess.run(self.local_Q.q_action, feed_dict={self.local_Q.inputs:np.vstack(next_observations),
                                                               self.local_Q.actions_q:action_next})
        q_target_batch = []
        for i in range(len(q_target)):
            qi = q_target[i]# * (1 - dones[i])
            z_target_step = []
            for j in range(len(qi)):
                z_target_step.append(gamma * qi[j] + rewards[i])
            q_target_batch.append(z_target_step)
        q_target_batch = np.array(q_target_batch)
        #print q_target_batch
        isweight = np.zeros((batch_size,N))
        for i in range(batch_size):
            for j in range(N):
                isweight[i,j] = ISWeights[i]
        feed_dict = {self.local_Q.inputs:np.vstack(observations),
                     self.local_Q.actions_q:action_now,
                     self.local_Q.q_target:q_target_batch,
                     self.local_Q.ISWeights:isweight}
        u,l,g_n,v_n,_ = sess.run([self.local_Q.u,
                                  self.local_Q.loss,
                                  self.local_Q.grad_norms,
                                  self.local_Q.var_norms,
                                  self.local_Q.apply_grads],feed_dict=feed_dict)
        return l/len(rollout), g_n, v_n, Q_target, u

    def work(self,gamma,sess,coord,saver):
        global GLOBAL_STEP
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        epsilon = 0.2
        
        print ("Starting worker " + str(self.number))
        best_mean_episode_reward = -float('inf')
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                #episode_buffer = []
                episode_reward = 0
                episode_step_count = 0
                d = False
                s = self.env.reset()
                s = process_frame(s)
                if epsilon > 0.01:
                    epsilon = epsilon * 0.997
                while not d:
                    #self.env.render()
                    GLOBAL_STEP += 1
                    #Take an action using probabilities from policy network output.
                    if random.random() > epsilon:
                        a_dist_list = sess.run(self.local_Q.Q, feed_dict={self.local_Q.inputs:[s]})
                        a_dist = a_dist_list[0]
                        a = np.argmax(a_dist)
                    else:
                        a = random.randint(0, 5)
                    
                    s1, r, d, _ = self.env.step(a)
                    if d == False:
                        s1 = process_frame(s1)
                    else:
                        s1 = s
                    self.replaymemory.add([s,a,r,s1,d])
                    episode_reward += r
                    s = s1                    
                    total_steps += 1
                    episode_step_count += 1
                    if total_steps % 2 == 0 and d != True and total_steps > 50000:
                        episode_buffer, tree_idx, ISWeights = self.replaymemory.sample(batch_size)
                        l,g_n,v_n,Q_target,u = self.train(episode_buffer,sess,gamma,ISWeights)
                        u = np.mean(u,axis=1)
                        self.replaymemory.update_priorities(tree_idx,u)
                        sess.run(self.update_local_ops)
                    if d == True:
                        break
                    
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)

                # Periodically save gifs of episodes, model parameters, and summary statistics.
                if episode_count % 5 == 0 and episode_count != 0 and total_steps > max_memory:
                    if self.name == 'worker_0' and episode_count % 5 == 0:
                        print('\n episode: ', episode_count, 'global_step:', \
                              GLOBAL_STEP, 'mean episode reward: ', np.mean(self.episode_rewards[-10:]), \
                              'epsilon: ', epsilon)
                    
                    print ('loss', l, 'Qtargetmean', np.mean(Q_target))
                    #print 'p_target', p_target
                    #print 'm', m
                    if episode_count % 100 == 0 and self.name == 'worker_0' and total_steps > 10000:
                        saver.save(sess,self.model_path+'/qr-dqn-'+str(episode_count)+'.cptk')
                        print ("Saved Model")

                    mean_reward = np.mean(self.episode_rewards[-100:])
                    if episode_count > 20 and best_mean_episode_reward < mean_reward:
                        best_mean_episode_reward = mean_reward

                if self.name == 'worker_0':
                    sess.run(self.increment)
                    #if episode_count%1==0:
                        #print('\r {} {}'.format(episode_count, episode_reward),end=' ')
                episode_count += 1

def get_env(task):
    env_id = task.env_id
    env = gym.make(env_id)
    env = wrap_deepmind(env)
    return env

gamma = 0.99 # discount rate for advantage estimation and reward discounting
s_size = 7056 # Observations are greyscale frames of 84 * 84 * 1
load_model = False
N = 20
k = 1.
model_path = 'a'
 # Get Atari games.
benchmark = gym.benchmark_spec('Atari40M')
# Change the index to select a different game.
task = benchmark.tasks[3]

tf.reset_default_graph()

if not os.path.exists(model_path):
    os.makedirs(model_path)
    
env = get_env(task)
a_size = env.action_space.n

batch_size = 32
global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)
trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
master_network = Q_Network(s_size,a_size,'global',None) # Generate global network
num_workers = 1 # Set workers ot number of available CPU threads
workers = []
max_memory = 50000
#replaymemory = ReplayMemory(max_memory)

# Create worker classes
for i in range(num_workers):
    #env = get_env(task)
    workers.append(Worker(env,i,s_size,a_size,trainer,model_path,global_episodes))
saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    sess.run(tf.global_variables_initializer())
    if load_model == True:
        print ('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess,ckpt.model_checkpoint_path)
        
    # This is where the asynchronous magic happens.
    # Start the "work" process for each worker in a separate threat.
    worker_threads = []
    for worker in workers:
        worker_work = lambda: worker.work(gamma,sess,coord,saver)
        t = threading.Thread(target=(worker_work))
        t.start()
        sleep(0.5)
        worker_threads.append(t)
    coord.join(worker_threads)
