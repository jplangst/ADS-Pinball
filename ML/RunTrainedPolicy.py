    # The code below can be used to run the trained policy when satisfied with the training results
    # NOTE that it needs to be updated to function with the pinball machine environment
    # Run trained policy
    #env = gym.make(ENVIRONMENT)
    #env = wrappers.Monitor(env, os.path.join(SUMMARY_DIR, ENVIRONMENT + "_trained"), video_callable=None)
    #while True:
    #    s = env.reset()
    #    ep_r, ep_t = 0, 0
    #    lstm_state = ppo.sess.run([ppo.pi_eval_i_state, ppo.vf_eval_i_state])
    #    while True:
    #        env.render()
    #        a, v, lstm_state = ppo.evaluate_state(s, lstm_state, stochastic=False)
    #        if not ppo.discrete:
    #            a = np.clip(a, env.action_space.low, env.action_space.high)
    #        s, r, terminal, _ = env.step(a)
    #        ep_r += r
    #        ep_t += 1
    #        if terminal:
    #            print("Reward: %.2f" % ep_r, '| Steps: %i' % ep_t)
    #            break