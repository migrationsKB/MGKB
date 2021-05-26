import pylode
from pylode.common import MakeDocco

def main():
    f = 'input/migrationsKB_schema.ttl'
    print('making ontdoc for {}'.format(f))
    h = MakeDocco(input_data_file=f)
    h.document(destination=f.replace('.ttl', '.html'))
    h = MakeDocco(input_data_file=f, outputformat="md")
    h.document(destination=f.replace(".ttl", ".md"))

if __name__ == '__main__':
    main()