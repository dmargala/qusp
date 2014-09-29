from collections import namedtuple
from operator import itemgetter

class Target(namedtuple('Target',('plate','mjd','fiber'))):
    def __str__(self):
        return '%d-%d-%d' % (self.plate,self.mjd,self.fiber)
    @classmethod
    def fromString(cls, targetString):
        plate, mjd, fiber = targetString.split('-')
        return cls(int(plate), int(mjd), int(fiber))

def readTargetList(filename, fields=[]):
    namedFields = [('plate',int,0),('mjd',int,0),('fiber',int,0)] + fields
    name, dtype, index = zip(*namedFields)

    class Target(namedtuple('Target',name)):
        def __str__(self):
            return '%d-%d-%d' % (self.plate,self.mjd,self.fiber)
        def attrs(self):
            return self[3:]

    targetList = []
    with open(filename,'r') as infile:
        for line in infile:
            tokens = itemgetter(0,*index[3:])(line.strip().split())
            namedTokens = tokens[0].split('-') + list(tokens)[1:]
            for i in range(len(namedTokens)):
                namedTokens[i] = dtype[i](namedTokens[i])
            targetList.append(Target(*namedTokens))
    return targetList

def saveTargetList(filename, targets):
    with open(filename, 'w') as outfile:
        for target in targets:
            outfile.write(' '.join([str(target)] + [str(attr) for attr in target.attrs()])+'\n')
            
