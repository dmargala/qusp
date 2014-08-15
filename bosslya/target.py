class Target(object):
    def __init__(self, plate, mjd, fiber):
        self.plate = int(plate)
        self.mjd = int(mjd)
        self.fiber = int(fiber)

    @classmethod
    def fromString(cls, targetString):
        targetString = targetString.strip().split()[0]
        plate, mjd, fiber = targetString.split('-')
        return cls(plate, mjd, fiber);

    def __str__(self):
        return '%d-%d-%d' % (self.plate,self.mjd,self.fiber)