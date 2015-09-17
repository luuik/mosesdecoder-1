#!/usr/bin/env perl

use strict;
use warnings;
use FindBin qw($RealBin);

binmode(STDIN, ":utf8");
binmode(STDOUT, ":utf8");

my $arpaPath = $ARGV[0];
my $outDir = $ARGV[1];

my $cmd = "mkdir -p $outDir";
print STDERR "Executing: $cmd\n";
`$cmd`;

open(ARPA_HANDLE, "<:encoding(UTF-8)", "$arpaPath") or die("Could not open file $arpaPath");
    	
my $unsortedFileName = "$outDir/tmp.$$.unsorted";
open(my $unsorted, ">:encoding(UTF-8)", "$unsortedFileName") or die("Could not open file $unsortedFileName");

my $vocabPathUnsorted = "$outDir/tmp.$$.vocab.dat.unsorted";
open(my $vocabUnsorted, ">:encoding(UTF-8)", "$vocabPathUnsorted") or die("Could not open file $vocabPathUnsorted");

while(my $line = <ARPA_HANDLE>) {
	chomp($line);
	my @toks = split('\t', $line);
  #print "line=$line \n";
	
	my $prob;
	my $backoff = 0;
	if (scalar(@toks) >= 2) {
	  # lm record
	  $prob = $toks[0];
	  
	  if (scalar(@toks) == 3) {
	  	# backoff
	  	$backoff = $toks[2];
	  }
	  
	  $line = $toks[1] ." $prob $backoff";
	  print $unsorted "$line\n";

	  # vocab
	  my @words = split(' ', $toks[1]);
	  for (my $i = 0; $i < scalar(@words); ++$i) {
	    print $vocabUnsorted $words[$i] ."\n";
  	}

	}		
}
close($unsorted);
close(ARPA_HANDLE);
close($vocabUnsorted);

# vocab
$cmd = "LC_ALL=C sort $vocabPathUnsorted | LC_ALL=C uniq > $outDir/vocab.dat";
print STDERR "Executing: $cmd\n";
`$cmd`;

# sort
my $sortedFileName = "$outDir/tmp.$$.sorted";
$cmd = "LC_ALL=C sort $unsortedFileName > $sortedFileName";
print STDERR "Executing: $cmd\n";
`$cmd`;

$cmd = "$RealBin/../../bin/processMorphoLMBin $sortedFileName $outDir";
print STDERR "Executing: $cmd\n";
`$cmd`;

unlink $unsortedFileName, $vocabPathUnsorted, $sortedFileName;

