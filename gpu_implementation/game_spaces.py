import gym_tensorflow
from pathlib import Path

gamelist = """
adventure.bin air_raid.bin alien.bin amidar.bin assault.bin asterix.bin asteroids.bin atlantis.bin bank_heist.bin battle_zone.bin beam_rider.bin berzerk.bin bowling.bin boxing.bin breakout.bin carnival.bin centipede.bin chopper_command.bin crazy_climber.bin defender.bin demon_attack.bin double_dunk.bin elevator_action.bin enduro.bin fishing_derby.bin freeway.bin frostbite.bin gopher.bin gravitar.bin hero.bin ice_hockey.bin jamesbond.bin journey_escape.bin kaboom.bin kangaroo.bin krull.bin kung_fu_master.bin montezuma_revenge.bin ms_pacman.bin name_this_game.bin phoenix.bin pitfall.bin pong.bin pooyan.bin private_eye.bin qbert.bin riverraid.bin road_runner.bin robotank.bin seaquest.bin skiing.bin solaris.bin space_invaders.bin star_gunner.bin tennis.bin time_pilot.bin tutankham.bin up_n_down.bin venture.bin video_pinball.bin wizard_of_wor.bin yars_revenge.bin zaxxon.bin
"""

games = [g.replace('.bin', '') for g in gamelist.split()]
skip_games = ['adventure', 'defender', 'kaboom']
games = set(games) - set(skip_games)

for game in games:
    env = gym_tensorflow.make(game, batch_size=8)
    print(game, env.action_space)
#env = gym.make('CartPole-v0')
#env.reset()
#for _ in range(1000):
#    env.render()
#    env.step(env.action_space.sample()) # take a random action
#env.close()
