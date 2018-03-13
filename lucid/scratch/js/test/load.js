const chai = require('chai');
const expect = chai.expect;

const fetchMock = require('fetch-mock');
const load = require("../public/index.js").load;

const mockData = { 'test': 42 }
fetchMock.mock('404', 404);
fetchMock.mock('test.json', JSON.stringify(mockData));
fetchMock.mock('test.csv', `NAME,VALUE,COLOR,DATE
Alan,12,blue,Sep. 25 2009
Minna,27,teal,Sep. 30 2009`);
fetchMock.mock('test.tsv', `NAME	VALUE	COLOR	DATE
Alan	12	blue	Sep. 25, 2009
Minna	27	teal	Sep. 30, 2009`);

describe('load()', function () {
  it('should load a json file', function (done) {
    load('test.json').then(result => {
      expect(JSON.stringify(result)).to.equal(JSON.stringify(mockData));
      done();
    });
  });
});