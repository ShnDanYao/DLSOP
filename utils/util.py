import logging
from logging import handlers




def sec_to_hm(t):
    """Convert time in seconds to times in hours, minutes and seconds
    e.g. 10239 -> (2, 50, 39)
    """
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return t, m, s

def sec_to_hm_to_string(t):
    """Conver time in seconds to a nice string
    e.g. 10239 -> 02h50m39s
    """
    h,m,s = sec_to_hm(t)
    if h==0:
        return "{:02d}m{:02d}s".format(m,s)
    else:
        return "{:02d}h{:02d}m{:02d}s".format(h,m,s)


class Logger(object):
    level_relations = {
            'debug':logging.DEBUG,
            'info':logging.INFO,
            'warning':logging.WARNING,
            'error':logging.ERROR,
            'crit':logging.CRITICAL
            }

    def __init__(self,filename,level='info',when='D',backCount=3,fmt=
            '%(message)s %(asctime)s',datefmt='%Y-%m-%d %I:%M:%S'):

        self.now=logging.getLogger(filename)
        format_str = logging.Formatter(fmt,datefmt=datefmt)
        self.format_str = format_str
        self.now.setLevel(self.level_relations.get(level))
        sh = logging.StreamHandler()
        sh.setFormatter(format_str)
        th = handlers.TimedRotatingFileHandler(filename,when=when,backupCount=backCount,encoding='utf-8')
        th.setFormatter(format_str)
        self.now.addHandler(sh)
        self.now.addHandler(th)

    def no_extra(self):
        format_str = logging.Formatter('%(message)s')
        self.now.handlers[0].setFormatter(format_str)
        self.now.handlers[1].setFormatter(format_str)

    def with_extra(self):
        self.now.handlers[0].setFormatter(self.format_str)
        self.now.handlers[1].setFormatter(self.format_str)

if __name__ == '__main__':
    log = Logger('all.log',level='debug')
    tests = 40
    log.now.debug('debug%d',tests)
    log.now.info('info')
    log.now.warning('警告')
    Logger('error.log', level='error').now.error('error')




