import numpy as np
from gym import spaces
import matplotlib.pyplot as plt
from scipy import stats

from recsim import document
from recsim import user
from recsim.choice_model import MultinomialLogitChoiceModel
from recsim.simulator import environment
from recsim.simulator import recsim_gym
import random
from absl import logging
from recsim.environments import interest_evolution
from recsim.agents import *


class EEVideo(document.AbstractDocument):
    # The maximum length of videos.
    MAX_VIDEO_LENGTH = 100.0
    # The number of features to represent each video
    NUM_FEATURES = 6

    def __init__(self,
                 doc_id,
                 features,
                 video_length=None,
                 pedagogy=None,
                 accuracy=None,
                 importance=None,
                 entertainment=None,
                 quality=None):
        """Generates a random set of features for this interest evolution Video."""

        # Document features (i.e. distribution over topics)
        self.features = features

        # Length of video
        self.video_length = video_length

        # Video evaluation measures considered as observable properties that the expert controls
        self.pedagogy = pedagogy
        self.accuracy = accuracy
        self.importance = importance
        self.entertainment = entertainment

        # Video quality
        self.quality = quality

        # doc_id is an integer representing the unique ID of this document
        super(EEVideo, self).__init__(doc_id)

    def create_observation(self):
        """Returns observable properties of this document as a float array."""
        f = ["f" + str(i) for i in range(self.NUM_FEATURES)]
        a = "pedagogy: " + str(self.pedagogy) + ", importance: " + str(self.importance) + ", accuracy: " \
            + str(self.accuracy) + ", entertainment: " + str(self.entertainment)
        doc_features = [self.pedagogy, self.importance, self.accuracy, self.entertainment]
        # return f, self.features, a
        return np.append(self.features, doc_features)
        #return self.features

    @classmethod
    def observation_space(cls):
        # Make sure of the dimension of the action space you are using
        return spaces.Box(
            shape=(cls.NUM_FEATURES + 4,), dtype=np.float32, low=-1.0, high=1.0)

    def __str__(self):
        return "Video {} with features {} and pedagogy {} and accuracy{} and importance {} " \
               "and entertainment{} and  video length{} and quality{}  ."\
            .format(self._doc_id, self.features, self.pedagogy, self.accuracy,
                                                          self.importance, self.entertainment, self.video_length,
                                                          self.quality)


class EEVideoSampler(document.AbstractDocumentSampler):
    def __init__(self, topic_id,
                 doc_ctor=EEVideo,
                 min_feature_value=-1.0,
                 max_feature_value=1.0,
                 video_length_mean=4.3,
                 video_length_std=1.0,
                 **kwargs):
        super(EEVideoSampler, self).__init__(doc_ctor, **kwargs)
        self._doc_count = 0
        self._min_feature_value = min_feature_value
        self._max_feature_value = max_feature_value
        self._video_length_mean = video_length_mean
        self._video_length_std = video_length_std
        self.topic_id = topic_id

    def sample_document(self):
        doc_features = {}
        doc_features['doc_id'] = self._doc_count
        # For now, assume the document properties are fixed by the experiment
        # It will probably make more sense to concentrate the interests around a few
        # (e.g. 5?) categories or have a more sophisticated generative process?
        # topic_id = self._rng.randint(0, self.get_doc_ctor().NUM_FEATURES)

        doc_features['features'] = np.zeros(self.get_doc_ctor().NUM_FEATURES)
        doc_features['features'][self.topic_id] = 1
        doc_features['video_length'] = min(
            self._rng.normal(self._video_length_mean, self._video_length_std),
            self.get_doc_ctor().MAX_VIDEO_LENGTH)
        # Evaluation video features are uniformly distributed over [-1,1]
        doc_features['pedagogy'] = np.random.uniform(-1, 1, 1)
        doc_features['accuracy'] = np.random.uniform(-1, 1, 1)
        doc_features['importance'] = np.random.uniform(-1, 1, 1)
        doc_features['entertainment'] = np.random.uniform(-1, 1, 1)

        # quality calculation based on equal weighting of quality metrics
        doc_features['quality'] = (doc_features['pedagogy'] + doc_features['accuracy']
                                   + doc_features['importance'] + doc_features['entertainment']) / 4

        self._doc_count += 1
        return self._doc_ctor(**doc_features)



class ExpertUserState(user.AbstractUserState):
    NUM_FEATURES = 6

    def __init__(self, net_accuracy_exposure,
                 net_pedagogy_exposure,
                 net_importance_exposure,
                 net_entertainment_exposure,
                 pedagogy_mean,
                 pedagogy_stddev,
                 accuracy_mean,
                 accuracy_stddev,
                 importance_mean,
                 importance_stddev,
                 entertainment_mean,
                 entertainment_stddev,
                 memory_discount,
                 sensitivity,
                 innovation_stddev,
                 user_interests,
                 observation_noise_stddev=0.1,
                 score_scaling=None,
                 attention_prob=None,
                 no_click_mass=None,
                 keep_interact_prob=None,
                 min_doc_utility=None,
                 watched_videos=None,
                 impressed_videos=None,
                 evaluated_videos=None,
                 step_penalty=None,
                 min_normalizer=None,
                 user_quality_factor=None,
                 user_update_alpha=None,
                 document_quality_factor=None,
                 time_budget=None,
                 ):
        #### User features

        # Exposure of the evaluation's measures to the expert throughout the videos he watches
        # State variables/Latent features
        self.net_accuracy_exposure = net_accuracy_exposure
        self.net_pedagogy_exposure = net_pedagogy_exposure
        self.net_importance_exposure = net_importance_exposure
        self.net_entertainment_exposure = net_entertainment_exposure

        # The user's interests (1 = very interested, -1 = disgust)
        # Another option could be to represent in [0,1] e.g. by dirichlet
        self.user_interests = user_interests

        # Amount of time in minutes this user has left in session.
        self.time_budget = time_budget

        # Probability of interacting with another element on the same slate
        self.keep_interact_prob = keep_interact_prob

        # Min utility to interact with a document
        self.min_doc_utility = min_doc_utility

        # Convenience wrapper
        self.choice_features = {
            'score_scaling': score_scaling,
            # Factor of attention to give for subsequent items on slate
            # Item i on a slate will get attention (attention_prob)^i
            'attention_prob': attention_prob,
            # Mass that user does not click on any item in the slate
            'no_click_mass': no_click_mass,
            # If using the multinomial proportion model with negative scores, this
            # negative value will be subtracted from all scores to make a valid
            # distribution for sampling.
            'min_normalizer': min_normalizer
        }
        ## Engagement parameters
        self.pedagogy_mean = pedagogy_mean
        self.pedagogy_stddev = pedagogy_stddev
        self.importance_mean = importance_mean
        self.importance_stddev = importance_stddev
        self.accuracy_mean = accuracy_mean
        self.accuracy_stddev = accuracy_stddev
        self.entertainment_mean = entertainment_mean
        self.entertainment_stddev = entertainment_stddev

        ## Transition model parameters
        ##############################

        # Step size for updating user interests based on watched videos (small!)
        # We may want to have different values for different interests
        # to represent how malleable those interests are (e.g. strong dislikes may
        # be less malleable).
        self.user_update_alpha = user_update_alpha

        # A step penalty applied when no item is selected (e.g. the time wasted
        # looking through a slate but not clicking, and any loss of interest)
        self.step_penalty = step_penalty

        # How much to weigh the user quality when updating budget
        self.user_quality_factor = user_quality_factor
        # How much to weigh the document quality when updating budget
        self.document_quality_factor = document_quality_factor

        # Examples of Observable user features
        ###########################

        # Video IDs of videos that have been watched
        self.watched_videos = watched_videos

        # Video IDs of videos that have been impressed
        self.impressed_videos = impressed_videos

        # Video IDs of evaluated videos
        self.evaluated_videos = evaluated_videos

        self.memory_discount = memory_discount
        self.sensitivity = sensitivity
        self.innovation_stddev = innovation_stddev

        ### State variables

        self.net_quality_exposure = 0.25 * net_accuracy_exposure + 0.25 * net_entertainment_exposure \
                                    + 0.25 * net_pedagogy_exposure + 0.25 * net_importance_exposure

        self.satisfaction = 1 / (1 + np.exp(-sensitivity * self.net_quality_exposure))
        self.time_budget = time_budget

        # Noise
        self._observation_noise = observation_noise_stddev

    def create_observation(self):
        """User's state is partially observable, it's noisy"""
        clip_low, clip_high = (-1.0 / (1.0 * self._observation_noise),
                               1.0 / (1.0 * self._observation_noise))
        noise = stats.truncnorm(
            clip_low, clip_high, loc=0.0, scale=self._observation_noise).rvs()
        noisy_sat = self.satisfaction * noise
        return np.append(np.array([noisy_sat]), self.user_interests)
        #return np.array([noisy_sat, ]), self.user_interests
        #return self.user_interests

    @classmethod
    def observation_space(cls):
        return spaces.Box(
            shape=(cls.NUM_FEATURES + 1,), dtype=np.float32, low=-1.0, high=1.0)

    # scoring function for use in the choice model -- the expert is more likely to
    # click on more correct/relevant content.
    def score_document(self, doc_obs):
        # print("doc_obs: ", doc_obs)
        # if self.user_interests.shape != doc_obs.shape:
        #     raise ValueError('User and document feature dimension mismatch!')
        scores_expert_interest = np.dot(self.user_interests, doc_obs[:self.NUM_FEATURES])
        q = sum(doc_obs[self.NUM_FEATURES:]) / self.NUM_FEATURES
        return 1 / (1 + np.exp(-scores_expert_interest * q))
        # return np.dot(self.user_interests, doc_obs)


class ExpertStaticUserSampler(user.AbstractUserSampler):
    _state_parameters = None

    def __init__(self,
                 user_ctor=ExpertUserState,
                 memory_discount=0.9,
                 sensitivity=0.01,
                 innovation_stddev=0.05,
                 time_budget=60,
                 document_quality_factor=1.0,
                 user_quality_factor=0.8,
                 no_click_mass=1.0,
                 min_normalizer=-1.0,
                 score_scaling=0.5,
                 step_penalty=0.05,
                 attention_prob=0.65,
                 pedagogy_mean=0,
                 pedagogy_stddev=0.2,
                 accuracy_mean=0,
                 accuracy_stddev=0.3,
                 importance_mean=0,
                 importance_stddev=0.3,
                 entertainment_mean=0,
                 entertainment_stddev=0.2,
                 user_update_alpha=0,
                 **kwargs):
        """Creates a new user state sampler."""
        self._state_parameters = {'memory_discount': memory_discount,
                                  'sensitivity': sensitivity,
                                  'innovation_stddev': innovation_stddev,
                                  'no_click_mass': no_click_mass,
                                  'min_normalizer': min_normalizer,
                                  'document_quality_factor': document_quality_factor,
                                  'time_budget': time_budget,
                                  'step_penalty': step_penalty,
                                  'score_scaling': score_scaling,
                                  'attention_prob': attention_prob,
                                  'pedagogy_mean': pedagogy_mean,
                                  'pedagogy_stddev': pedagogy_stddev,
                                  'accuracy_mean': accuracy_mean,
                                  'accuracy_stddev': accuracy_stddev,
                                  'importance_mean': importance_mean,
                                  'importance_stddev': importance_stddev,
                                  'entertainment_mean': entertainment_mean,
                                  'entertainment_stddev': entertainment_stddev,
                                  'user_quality_factor': user_quality_factor,
                                  'user_update_alpha': user_update_alpha
                                  }
        super(ExpertStaticUserSampler, self).__init__(user_ctor, **kwargs)

    def sample_user(self):
        self._state_parameters['net_accuracy_exposure'] = ((self._rng.random_sample() - .5) *
                                                           (1 / (1.0 - self._state_parameters['memory_discount'])))
        self._state_parameters['net_pedagogy_exposure'] = ((self._rng.random_sample() - .5) *
                                                           (1 / (1.0 - self._state_parameters['memory_discount'])))
        self._state_parameters['net_entertainment_exposure'] = ((self._rng.random_sample() - .5) *
                                                                (1 / (1.0 - self._state_parameters['memory_discount'])))
        self._state_parameters['net_importance_exposure'] = ((self._rng.random_sample() - .5) *
                                                             (1 / (1.0 - self._state_parameters['memory_discount'])))
        self._state_parameters['user_interests'] = self._rng.uniform(-1.0, 1.0, self.get_user_ctor().NUM_FEATURES)
        utility_range = 1.0 / 1.2
        # Fraction of video length we can extend (or cut) budget by
        alpha = 0.9
        self._state_parameters['user_update_alpha'] = alpha * utility_range

        return self._user_ctor(**self._state_parameters)


class ExpertResponse(user.AbstractResponse):
    # The maximum degree of engagement.
    MAX_ENGAGEMENT_MAGNITUDE = 100.0

    def __init__(self, clicked=False, accuracy_eval=False, pedagogy_eval=False, importance_eval=False, evaluated=False,
                 engagement_eval=False,  watch_time=0.0, engagement=0.0):
        self.clicked = clicked
        self.evaluated = evaluated
        self.accuracy_eval = accuracy_eval
        self.pedagogy_eval = pedagogy_eval
        self.importance_eval = importance_eval
        self.entertainment_eval = engagement_eval
        self.watch_time = watch_time
        self.engagement = engagement

    def create_observation(self):
        # return {'click': int(self.clicked), 'evaluation': int(self.evaluated),
        #         'accuracy_eval': float(self.accuracy_eval), 'pedagogy_eval': float(self.pedagogy_eval),
        #         'importance_eval': float(self.importance_eval), 'entertainment_eval': float(self.entertainment_eval),
        #         'watch_time': np.array(self.watch_time), 'engagement': np.array(self.engagement),
        #         'quality': float(self.quality),'cluster_id': int(self.cluster_id)}

        return np.array([int(self.clicked), int(self.evaluated), float(self.accuracy_eval),
                         float(self.pedagogy_eval), float(self.importance_eval), float(self.entertainment_eval), float(self.watch_time),
                         float(self.engagement)])

    @classmethod
    def response_space(cls):
        # `engagement` feature range is [0, MAX_ENGAGEMENT_MAGNITUDE]
        return spaces.Dict({
            'click':
                spaces.Discrete(2),
            'accuracy_eval':
                spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=tuple(),
                    dtype=np.float32),
            'pedagogy_eval':
                spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=tuple(),
                    dtype=np.float32),
            'importance_eval':
                spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=tuple(),
                    dtype=np.float32),
            'entertainment_eval':
                spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=tuple(),
                    dtype=np.float32),
            'watch_time':
                spaces.Box(
                    low=0.0,
                    high=EEVideo.MAX_VIDEO_LENGTH,
                    shape=tuple(),
                    dtype=np.float32),
            'evaluation':
                spaces.Discrete(2),
            'engagement':
                spaces.Box(
                    low=0.0,
                    high=cls.MAX_ENGAGEMENT_MAGNITUDE,
                    shape=tuple(),
                    dtype=np.float32),

        })


def user_init(self, size_slate, seed=0, no_click_mass=1.0, alpha_x_intercept=1.0,
              alpha_y_intercept=0.3):
    super(ExpertModel,
          self).__init__(ExpertResponse,
                         ExpertStaticUserSampler(ExpertUserState, no_click_mass=no_click_mass, seed=seed), size_slate)
    self.choice_model = MultinomialLogitChoiceModel({})
    self._alpha_x_intercept = alpha_x_intercept
    self._alpha_y_intercept = alpha_y_intercept
    self.scores = None


def simulate_response(self, documents):
    # List of empty responses
    responses = [self._response_model_ctor() for _ in documents]

    # Sample some clicked responses using user's choice model and populate responses.
    doc_obs = [doc.create_observation() for doc in documents]
    self.choice_model.score_documents(self._user_state, doc_obs)
    # print("doc_obs: ",doc_obs)
    self._scores = self.choice_model._scores
    selected_index = self.choice_model.choose_item()

    if selected_index is None:
        return responses
    self._generate_response(documents[selected_index],
                            responses[selected_index])
    return responses, self._scores


def generate_response(self, doc, response):
    x = random.random()
    # Considering this simple model of video evaluation
    response.clicked = True
    # 98% of states are evaluated
    if x <= 0.98:
        response.evaluated = True
        response.accuracy_eval = np.random.uniform(-1, 1, 1)
        response.importance_eval = np.random.uniform(-1, 1, 1)
        response.pedagogy_eval = np.random.uniform(-1, 1, 1)
        response.entertainment_eval = np.random.uniform(-1, 1, 1)
        doc.quality = (response.accuracy_eval + response.importance_eval
                       + response.pedagogy_eval + response.entertainment_eval) / 4
        doc.pedagogy = response.pedagogy_eval
        doc.importance = response.importance_eval
        doc.entertainment = response.entertainment_eval
        doc.accuracy = response.accuracy_eval

    engagement_loc = (doc.pedagogy * self._user_state.pedagogy_mean
                      + doc.accuracy * self._user_state.accuracy_mean + doc.importance
                      * self._user_state.importance_mean + doc.entertainment
                      * self._user_state.entertainment_mean) / 4

    engagement_loc *= self._user_state.satisfaction
    engagement_scale = (self._user_state.pedagogy_stddev
                        + self._user_state.accuracy_stddev
                        + self._user_state.importance_stddev
                        + self._user_state.entertainment_stddev) / 4
    log_engagement = np.random.normal(loc=engagement_loc,
                                      scale=engagement_scale)
    response.engagement = np.exp(log_engagement)
    response.watch_time = min(self._user_state.time_budget, doc.video_length)


def update_state(self, slate_documents, responses):
    for doc, response in zip(slate_documents, responses):
        # Step size should vary based on interest.
        def compute_alpha(x, x_intercept, y_intercept):
            return (-y_intercept / x_intercept) * np.absolute(x) + y_intercept

        if response.clicked:
            innovation = np.random.normal(scale=self._user_state.innovation_stddev)
            if response.evaluated:

                net_quality_exposure = (self._user_state.memory_discount
                                        * self._user_state.net_quality_exposure
                                        + doc.quality
                                        + innovation
                                        )
            else:

                net_quality_exposure = (self._user_state.memory_discount
                                        * self._user_state.net_quality_exposure
                                        - innovation
                                        )

            self._user_state.net_quality_exposure = net_quality_exposure
            self.choice_model.score_documents(
                self._user_state, [doc.create_observation()])
            # scores is a list of length 1 since only one doc observation is set.
            expected_utility = self.choice_model.scores[0]
            ## Update interests
            target = doc.features - self._user_state.user_interests
            mask = doc.features
            alpha = compute_alpha(self._user_state.user_interests,
                                  self._alpha_x_intercept, self._alpha_y_intercept)

            update = alpha * mask * target
            positive_update_prob = np.dot((self._user_state.user_interests + 1.0) / 2,
                                          mask)
            flip = np.random.rand(1)
            if flip < positive_update_prob:
                self._user_state.user_interests += update
            else:
                self._user_state.user_interests -= update
            self._user_state.user_interests = np.clip(self._user_state.user_interests, -1.0, 1.0)
            ## Update budget
            received_utility = (self._user_state.user_quality_factor * expected_utility) + (
                    self._user_state.document_quality_factor * float(doc.quality))
            self._user_state.time_budget -= response.watch_time
            self._user_state.time_budget += (response.watch_time * received_utility)
            satisfaction = 1 / (1.0 + np.exp(-self._user_state.sensitivity
                                             * net_quality_exposure))
            self._user_state.satisfaction = satisfaction
            # self._user_state.time_budget -= 1

            return
    # Step penalty if no selection
    self._user_state.time_budget -= self._user_state.step_penalty


def is_terminal(self):
    """Returns a boolean indicating if the session is over."""
    #print("time_budget ", self._user_state.time_budget)
    return self._user_state.time_budget <= 0


ExpertModel = type("ExpertModel", (user.AbstractUserModel,),
                   {"__init__": user_init,
                    "is_terminal": is_terminal,
                    "update_state": update_state,
                    "simulate_response": simulate_response,
                    "_generate_response": generate_response})

if __name__ == '__main__':

    def clicked_evaluation_reward(responses):
        reward = 0.0
        for response in responses:
            if response.clicked:
                if response.evaluated:
                    reward += int((response.accuracy_eval + response.importance_eval
                                   + response.pedagogy_eval + response.entertainment_eval) / 4)
        return reward

    def clicked_engagement_reward(responses):
        reward = 0.0
        for response in responses:
            if response.clicked:
                    reward += response.watch_time
        return reward

    slate_size = 2
    num_candidates = 5
    expertEnv = environment.Environment(
        ExpertModel(slate_size),
        EEVideoSampler(),
        num_candidates,
        slate_size,
        resample_documents=True)
    env_config = {
        'num_candidates': num_candidates,
        'slate_size': slate_size,
        'resample_documents': True,
        'seed': 0,
    }

    lts_gym_env = recsim_gym.RecSimGymEnv(expertEnv, clicked_evaluation_reward)
    recsim_gym_env = interest_evolution.create_environment(env_config)
    # observation_0 = lts_gym_env.reset()

    from recsim.agents import random_agent
    from recsim.agents import full_slate_q_agent
    from recsim.simulator import runner_lib


    def create_agent(sess, environment, eval_mode, summary_writer=None):
        """Creates an instance of FullSlateQAgent.

        Args:
          sess: A `tf.Session` object for running associated ops.
          environment: A recsim Gym environment.
          eval_mode: A bool for whether the agent is in training or evaluation mode.
          summary_writer: A Tensorflow summary writer to pass to the agent for
            in-agent training statistics in Tensorboard.

        Returns:
          An instance of FullSlateQAgent.
        """
        kwargs = {
            'observation_space': environment.observation_space,
            'action_space': environment.action_space,
            'summary_writer': summary_writer,
            'eval_mode': eval_mode,
        }
        return full_slate_q_agent.FullSlateQAgent(sess, **kwargs)


    # Create agent
    # action_space = spaces.MultiDiscrete(num_candidates * np.ones((slate_size,)))
    # agent = random_agent.RandomAgent(action_space, random_seed=0)
    # observation_1 = observation_0
    # print(observation_0['doc'])
    # for i in range(3):
    #     recommendation_slate_0 = agent.step(observation_1)
    #     print(recommendation_slate_0)
    #     observation_1, reward, done, scores, _ = recsim_gym_env.step(recommendation_slate_0)
    #     print('Observation ' + str(i))
    #     print('Available documents')
    #     doc_strings = ['doc_id ' + key + str(value) for key, value
    #                    in observation_1['doc'].items()]
    #     print('\n'.join(doc_strings))
    #     rsp_strings = [str(response) for response in observation_1['response']]
    #     print('User responses to documents in the slate')
    #     print('\n'.join(rsp_strings))
    #     print('Reward: ', reward)
    #     print("User observation noise:", observation_1['user'][0], " interests features: ", observation_1['user'][1:])
    #     print("*******************************************")

    tmp_base_dir = './results/expert_eval/'

    runner = runner_lib.TrainRunner(
        base_dir=tmp_base_dir,
        create_agent_fn=create_agent,
        env=lts_gym_env,
        episode_log_file='',
        max_training_steps=50,
        num_iterations=10)
    runner.run_experiment()

    runner = runner_lib.EvalRunner(
        base_dir=tmp_base_dir,
        create_agent_fn=create_agent,
        env=recsim_gym_env,
        max_eval_episodes=5,
        test_mode=True)
    runner.run_experiment()
