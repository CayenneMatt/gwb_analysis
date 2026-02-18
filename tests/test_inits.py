# This test will import important modules and libraries.

from gwb_analysis.analyze_model import Model_Info
import pytest

def test_model_info_posteriors():
	path = '/Users/cayenne/Documents/Research/amplitude_paper/all_data/'
	asa = Model_Info(path=path, file='ASA_noev.npz', model_name='Le11ne', color='b', line_style='-', threshold=2.5).get_posteriors()
	# pytest.approx(asa.posteriors['hard_time'], float(3.5841389851321646))
	assert asa.posteriors['hard_time'] == float(3.5841389851321646)


def test_read_fits():
	path = '/Users/cayenne/Documents/Research/amplitude_paper/all_data/'
	asa = Model_Info(path=path, file='ASA_noev.npz', model_name='Le11ne', color='b', line_style='-', threshold=2.5)
	fit = asa.get_shenf(0.2)
	assert fit[1][0] == float(-2.241516079748106)
