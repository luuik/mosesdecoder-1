#!/usr/bin/env perl

use strict;
use warnings;

binmode(STDIN, ":utf8");
binmode(STDOUT, ":utf8");

my $unsortedFileName = "tmp.$$.unsorted";
open(my $unsorted, ">$unsortedFileName") or die("Could not open file $unsortedFileName");

while(my $line = <STDIN>) {
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
	}	
}
close($unsorted);

# sort
my $sortedFileName = "tmp.$$.sorted";
my $cmd = "LC_ALL=C sort $unsortedFileName > $sortedFileName";
`$cmd`;
