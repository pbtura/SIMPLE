

def get_environment(env_name):
    try:
        if env_name in ('tictactoe'):
            from app.environments.tictactoe.tictactoe.envs.tictactoe import TicTacToeEnv
            return TicTacToeEnv
        elif env_name in ('connect4'):
            from app.environments.connect4.connect4.envs.connect4 import Connect4Env
            return Connect4Env
        elif env_name in ('sushigo'):
            from app.environments.sushigo.sushigo.envs.sushigo import SushiGoEnv
            return SushiGoEnv
        elif env_name in ('butterfly'):
            from app.environments.butterfly.butterfly.envs.butterfly import ButterflyEnv
            return ButterflyEnv
        elif env_name in ('geschenkt'):
            from app.environments.geschenkt.geschenkt.envs.geschenkt import GeschenktEnv
            return GeschenktEnv
        elif env_name in ('frouge'):
            from app.environments.frouge.frouge.envs.frouge import FlammeRougeEnv
            return FlammeRougeEnv
        else:
            raise Exception(f'No environment found for {env_name}')
    except SyntaxError as e:
        print(e)
        raise Exception(f'Syntax Error for {env_name}!')
    except:
        raise Exception(f'Install the environment first using: \nbash scripts/install_env.sh {env_name}\nAlso ensure the environment is added to /utils/register.py')
    


def get_network_arch(env_name):
    if env_name in ('tictactoe'):
        from app.models.tictactoe.models import CustomPolicy
        return CustomPolicy
    elif env_name in ('connect4'):
        from app.models.connect4.models import CustomPolicy
        return CustomPolicy
    elif env_name in ('sushigo'):
        from app.models.sushigo.models import CustomPolicy
        return CustomPolicy
    elif env_name in ('butterfly'):
        from app.models.butterfly.models import CustomPolicy
        return CustomPolicy
    elif env_name in ('geschenkt'):
        from app.models.geschenkt.models import CustomPolicy
        return CustomPolicy
    elif env_name in ('frouge'):
        from app.models.frouge.models import CustomPolicy
        return CustomPolicy
    else:
        raise Exception(f'No model architectures found for {env_name}')

