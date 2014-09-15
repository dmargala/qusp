from collections import namedtuple

class Target(namedtuple('Target',('plate','mjd','fiber'))):
    def __str__(self):
        return '%d-%d-%d' % (self.plate,self.mjd,self.fiber)
    @classmethod
    def fromString(cls, targetString):
        plate, mjd, fiber = targetString.split('-')
        return cls(int(plate), int(mjd), int(fiber))

def readTargetList(filename, fields=[]):
    namedFields = [('plate',int),('mjd',int),('fiber',int)] + fields
    name, dtype = zip(*namedFields)

    class Target(namedtuple('Target',name)):
        def __str__(self):
            return '%d-%d-%d' % (self.plate,self.mjd,self.fiber)
        def attrs(self):
            return self[3:]

    targetList = []
    with open(filename,'r') as infile:
        for line in infile:
            tokens = line.strip().split()
            namedTokens = tokens[0].split('-') + tokens[1:1+len(fields)]
            for i in range(len(namedTokens)):
                namedTokens[i] = dtype[i](namedTokens[i])
            targetList.append(Target(*namedTokens))
    return targetList

