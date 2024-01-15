#!-*- coding: utf-8 -*-

import utils

class Settings:

    default = {
        'basis'     : 'configuration',
        'excitation': 'cis',
    }

    
    @classmethod
    def load_setting(cls, settings, key):
    
        if type(key) is str:
    
            try:
                
                return settings[key]
    
            except KeyError:
                
                return cls.default[key]
    
        elif hasattr(key, "__iter__"):
    
            if len(key) > 1:
            
                return cls.load_setting(settings[key[0]], key[1:])
    
            elif len(key) == 1:
    
                return cls.load_setting(settings, key[-1])
    
            else:
    
                utils.stop_with_error("Invalid key.\n")
    
        else:
            
            utils.stop_with_error("Invalid key.\n")
    
