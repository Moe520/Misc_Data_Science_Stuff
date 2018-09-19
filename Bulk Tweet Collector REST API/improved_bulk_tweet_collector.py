# Huge thanks to Mark Dredze https://github.com/mdredze for the base code


'''
Gets text content for tweet IDs
'''

# standard
from __future__ import print_function
import getopt
import logging
import os
import sys
# import traceback
# third-party: `pip install tweepy`
import tweepy

# global logger level is configured in main()
Logger = None

# Generate your own at https://apps.twitter.com/app
CONSUMER_KEY = ''
CONSUMER_SECRET = ''
OAUTH_TOKEN = ''
OAUTH_TOKEN_SECRET = ''

def get_tweet_id(line):
    '''
    Extracts and returns tweet ID from a line in the input.
    '''
    (tweet_id) = line
    return tweet_id

def get_tweets_single(twapi, idfilepath):
    '''
    Fetches content for tweet IDs in a file one at a time,
    which means a ton of HTTPS requests, so NOT recommended.

    `twapi`: Initialized, authorized API object from Tweepy
    `idfilepath`: Path to file containing IDs
    '''
    
    # process IDs from the file
    with open(idfilepath, 'rb') as idfile:
        for line in idfile:
            tweet_id = get_tweet_id(line)
            Logger.debug('Fetching tweet for ID %s', tweet_id)
            try:
                tweet = twapi.get_status(tweet_id)
                print('%s,%s' % (tweet_id, tweet.text.encode('UTF-8')))
            except tweepy.TweepError as te:
                Logger.warn('Failed to get tweet ID %s: %s', tweet_id, te.message)
                # traceback.print_exc(file=sys.stderr)
        # for
    # with

def get_tweet_list(twapi, idlist):
    '''
    Invokes bulk lookup method.
    Raises an exception if rate limit is exceeded.
    '''
    # fetch as little metadata as possible
    tweets = twapi.statuses_lookup(id_=idlist, include_entities=False, trim_user=False)
    for tweet in tweets:
        print('%s,%s' % (tweet.id, tweet.text.encode('UTF-8')))
        with open('output.txt', 'a') as outfile:
            outfile.write('%s,"%s",%s ,%s,%s,%s,%s,"%s",%s' % (tweet.id , tweet.text.encode('UTF-8'),tweet.created_at ,tweet.user.followers_count,tweet.user.friends_count,tweet.user.statuses_count,tweet.user.favourites_count,tweet.user.location.encode('UTF-8'),'\n'))
        

def get_tweets_bulk(twapi, idfilepath):
    '''
    Fetches content for tweet IDs in a file using bulk request method,
    which vastly reduces number of HTTPS requests compared to above;
    however, it does not warn about IDs that yield no tweet.

    `twapi`: Initialized, authorized API object from Tweepy
    `idfilepath`: Path to file containing IDs
    '''
    # process IDs from the file
    tweet_ids = list()
    with open(idfilepath, 'rb') as idfile:
        for line in idfile:
            tweet_id = get_tweet_id(line)
            Logger.debug('Fetching tweet for ID %s', tweet_id)
            # API limits batch size to 100
            if len(tweet_ids) < 100:
                tweet_ids.append(tweet_id)
            else:
                get_tweet_list(twapi, tweet_ids)
                tweet_ids = list()
    # process rump of file
    if len(tweet_ids) > 0:
        get_tweet_list(twapi, tweet_ids)

def usage():
    print('Usage: get_tweets_by_id.py [options] file')
    print('    -s (single) makes one HTTPS request per tweet ID')
    print('    -v (verbose) enables detailed logging')
    sys.exit()

def main(args):
    logging.basicConfig(level=logging.WARN)
    global Logger
    Logger = logging.getLogger('get_tweets_by_id')
    bulk = True
    try:
        opts, args = getopt.getopt(args, 'sv')
    except getopt.GetoptError:
        usage()
    for opt, _optarg in opts:
        if opt in ('-s'):
            bulk = False
        elif opt in ('-v'):
            Logger.setLevel(logging.DEBUG)
            Logger.debug("verbose mode on")
        else:
            usage()
    if len(args) != 1:
        usage()
    idfile = args[0]
    if not os.path.isfile(idfile):
        print('Not found or not a file: %s' % idfile, file=sys.stderr)
        usage()

    # connect to twitter
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
    api = tweepy.API(auth)

    # hydrate tweet IDs
    if bulk:
        get_tweets_bulk(api, idfile)
    else:
        get_tweets_single(api, idfile)

if __name__ == '__main__':
    main(sys.argv[1:])
